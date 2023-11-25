from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import pickle
# from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Yu Gothic']

# import cal_acc
'''
'objective' : 'regression',
'objective' : 'binary', 
'metric': {'mae'},
'metric': {'mse'},
'metric': {'binary_logloss'},
'metric': {'binary_error'},  評価指標 : 誤り率(= 1-正答率)
'''


class LrSchedulingCallback(object):
    def __init__(self):
        # 検証用データに対する評価指標の履歴
        self.eval_metric_history = []
        # 評価指標
        self.ave_latest_before = float('inf')
        self.cal_ave_latest_before = True
    
    def __call__(self, env):
        # 現在の学習率を取得する
        current_lr = env.params.get('learning_rate')

        # 検証用データに対する評価結果を取り出す（先頭の評価指標）
        first_eval_result = env.evaluation_result_list[1]
        # print(env.evaluation_result_list[0])
        # print(env.evaluation_result_list[1])

        # スコア
        metric_score = first_eval_result[2]
        # 評価指標は大きい方が優れているかどうか
        is_higher_better = first_eval_result[3]

        # 評価指標の履歴を更新する
        self.eval_metric_history.append(metric_score)

        # 最新の評価指標が向上していなければ学習率を更新
        new_lr = current_lr
        if len(self.eval_metric_history) >= 100:
            latest_hist = self.eval_metric_history[-100:]
            if self.cal_ave_latest_before:
                self.ave_latest_before = sum(latest_hist) / len(latest_hist)
                self.cal_ave_latest_before = False
            ave_latest = sum(latest_hist) / len(latest_hist)
            if is_higher_better:
                if ave_latest < self.ave_latest_before:
                    new_lr = current_lr * 0.8
                    self.ave_latest_before = ave_latest
                    self.eval_metric_history = []
            else:
                if ave_latest > self.ave_latest_before:
                    new_lr = current_lr * 0.8
                    self.ave_latest_before = ave_latest
                    self.eval_metric_history = []

        # 学習率の下限
        min_threshhold = 0.0001
        new_lr = max(min_threshhold, new_lr)

        # 次のラウンドで使う学習率を更新
        update_params = {'learning_rate' : new_lr}
        env.model.reset_parameter(update_params)
        env.params.update(update_params)
    
    @property
    def beforeiteration(self):
        # コールバックは各イテレーションの後に実行する
        return False



### LightGBMで学習 ###
def LightGBM_Grid(X_train, Y_train, n_class, esr, rate_list, depth_list, leaves_list, min_leaf_list, LGBM_seed, LGBM_feature_fraction, KF_splits):
    start_time = datetime.datetime.now()

    # グリッドサーチ総数
    total = len(rate_list) * len(depth_list) * len(leaves_list) * len(min_leaf_list)
    print('グリッドサーチパラメータ総数: {}'.format(total))

    # 交差検証の分割数
    kf = KFold(n_splits=KF_splits, shuffle=True, random_state=0)

    # LightGBMのグリッドサーチ
    best_score = float('-inf')
    best_parameters = {}
    count = 0
    for rate in rate_list:
        for depth in depth_list:
            for leaves in leaves_list:
                for min_leaf in min_leaf_list:
                    count += 1
                    # コールバック
                    lr_scheduler = LrSchedulingCallback()

                    # パラメータ
                    params = {'objective' : 'multiclass',
                              'num_class' : n_class, # 多クラスのクラス数を指定
                              'metric': {'multi_error'},
                              'early_stopping_rounds' : esr,   # early_stopping 回数指定
                              'learning_rate' : rate,
                              'max_depth' : depth,
                              'num_leaves' : leaves,
                              'min_data_in_leaf' : min_leaf,
                              'boosting_type' : 'gbdt',
                              'seed': LGBM_seed,
                              'feature_fraction' : LGBM_feature_fraction,
                              'force_col_wise' : True,
                              'verbose' : -1
                              }
                    
                    # トレーニングデータ」を学習用・検証用に分割
                    best_KF_score = 0
                    for _fold, (train_index, valid_index) in enumerate(kf.split(Y_train)):
                        X_trn = X_train[train_index]
                        Y_trn = Y_train[train_index]
                        X_val = X_train[valid_index]
                        Y_val = Y_train[valid_index]

                        # LightGBMの学習
                        lgb_dataset_trn = lgb.Dataset(X_trn, label=Y_trn, categorical_feature='auto')
                        lgb_dataset_val = lgb.Dataset(X_val, label=Y_val, categorical_feature='auto')

                        result_dic ={}
                        model = lgb.train(params = params, 
                                        train_set = lgb_dataset_trn, 
                                        valid_sets = [lgb_dataset_trn, lgb_dataset_val], 
                                        num_boost_round = 10000, 
                                        # early_stopping_rounds = esr, 
                                        # verbose_eval = -1,
                                        # evals_result = result_dic,
                                        callbacks = [lr_scheduler,
                                                     lgb.early_stopping(stopping_rounds=esr, verbose=True), # early_stopping用コールバック関数
                                                     lgb.record_evaluation(result_dic),
                                                     lgb.log_evaluation(-1)]
                                        )

                        # train_pred_prob = model.predict(X_trn, num_iteration=model.best_iteration)
                        # train_pred = np.where(train_pred_prob < 0.5, 0, 1) # 0.5より小さい場合0 ,そうでない場合1を返す
                        # train_pred = np.argmax(train_pred_prob, axis=1)
                        # train_acc = accuracy_score(Y_trn, train_pred)

                        val_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
                        val_pred = np.argmax(val_pred_prob, axis=1)
                        # val_acc = acc_list[3]
                        val_acc = accuracy_score(Y_val, val_pred)

                        # 学習率の取得
                        last_lr = model.params.get('learning_rate')

                        # 最も良いスコアのパラメータとスコアを更新
                        if val_acc > best_KF_score:
                            best_KF_score = val_acc
                            best_KF_model = model
                            best_KF_result = result_dic
                            best_KF_last_lr = last_lr
                    
                    # 最も良いスコアのパラメータとスコアを更
                    if best_KF_score > best_score:
                        best_score = best_KF_score
                        best_model = best_KF_model
                        best_result = best_KF_result
                        best_parameters = {'rate' : rate,
                                           'last_rate' : best_KF_last_lr,
                                           'depth' : depth,
                                           'leaves' : leaves,
                                           'min_leaf' : min_leaf
                                           }
                    
                    if count == 1 or count == int(total/5)*5 or count % (int(total/10)+1) == 0:
                        print('{}/{} Best score: {:.3f}%'.format(count, total, best_score))
                        print(datetime.datetime.now() - start_time)

    print()
    print('Best score: {:.3f}'.format(best_score))
    print('Best parameters: {}'.format(best_parameters))

    # 学習経過を表示
    fig = plt.figure(figsize=(11,7))
    plt.plot(best_result['training']['multi_error'], label='train')
    plt.plot(best_result['valid_1']['multi_error'], label='valid')
    plt.xlabel('num of iteration')
    plt.ylabel('multi error')
    plt.close()
    fig.savefig('LightGBM_learning.png', bbox_inches='tight')

    # モデルを保存
    with open('LightGBM_model.pickle', 'wb') as f:
        pickle.dump(best_model, f)
    
    # 予測
    Y_pred_prob = best_model.predict(X_train)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    df_lightGBM = pd.concat([pd.DataFrame(X_train), pd.DataFrame(Y_train)], axis=1)
    df_lightGBM['予測'] = Y_pred

    # 特徴量の重要度出力
    importance_gain = best_model.feature_importance(importance_type='gain')
    importance_rate = np.array(importance_gain) / np.sum(np.array(importance_gain)) * 100
    feature_importance = pd.DataFrame({'feature_name' : best_model.feature_name(),
                                       'importance' : importance_gain,
                                       'rate' : importance_rate
                                      })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(11,7))
    # sns.set(font='Yu Gothic')
    sns.barplot(data=feature_importance, x='importance', y='feature_name')
    plt.close()
    plt.savefig('feature_importance.png')

    calc_time = datetime.datetime.now() - start_time
    print(calc_time)

    return df_lightGBM, best_parameters, feature_importance, best_model



### LightGBMでOptuna ###
def LightGBM_Optuna(X_train, Y_train, n_class, esr, rate_list, depth_list, leaves_list, min_leaf_list, LGBM_seed, LGBM_feature_fraction, KF_splits):
    start_time = datetime.datetime.now()

    class Objective():
        def __init__(self, study, savedir):
            self.direction = study.direction
            self.best_score = None
            self.best_model = None
            self.directory_path = savedir
            self.best_result = None
        
        def __call__(self, trial):
            esr = 300
            kf = KFold(n_splits=5, shuffle=True, random_state=0)

            # パラメータ
            params = {'objective' : 'multiclass',
                      'num_class' : n_class, # 多クラスのクラス数を指定
                      'metric': {'multi_error'},
                      'early_stopping_rounds' : esr,   # early_stopping 回数指定
                      'learning_rate' : trial.suggest_uniform('learning_rate', 0.001, 0.2),
                      'max_depth' : trial.suggest_int('max_depth', 2, 20),
                      'num_leaves' : trial.suggest_int('num_leaves', 2, 50),
                      'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 1, 10),
                      'lambda_l1' : trial.suggest_loguniform('lambda_l1', 1e-3, 10.0),
                      'lambda_l2' : trial.suggest_loguniform('lambda_l2', 1e-3, 10.0),
                      'bagging_fraction' : trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
                      'bagging_freq' : trial.suggest_int('bagging_freq', 1, 5),
                      'seed': 2023,
                      'feature_fraction' : 1.0,
                      'force_col_wise' : True,
                      'verbose' : -1
                      }
            
            # トレーニングデータ」を学習用・検証用に分割
            best_KF_score = 0
            for _fold, (train_index, valid_index) in enumerate(kf.split(Y_train)):
                X_trn = X_train[train_index]
                Y_trn = Y_train[train_index]
                X_val = X_train[valid_index]
                Y_val = Y_train[valid_index]

                # LightGBMの学習
                lgb_dataset_trn = lgb.Dataset(X_trn, label=Y_trn, categorical_feature='auto')
                lgb_dataset_val = lgb.Dataset(X_val, label=Y_val, categorical_feature='auto')

                result_dic ={}
                model = lgb.train(params = params, 
                                train_set = lgb_dataset_trn, 
                                valid_sets = [lgb_dataset_trn, lgb_dataset_val], 
                                num_boost_round = 10000, 
                                early_stopping_rounds = esr, 
                                verbose_eval = False,
                                evals_result = result_dic,
                                )

                val_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
                val_pred = np.argmax(val_pred_prob, axis=1)
                # val_acc = acc_list[3]
                val_acc = accuracy_score(Y_val.values, val_pred)

                # 最も良いスコアのパラメータとスコアを更新
                if val_acc > best_KF_score:
                    best_KF_score = val_acc
            
            if self.best_score is None:
                self.save_best_model(model, best_KF_score)
            
            if best_KF_score > self.best_score:
                self.save_best_model(model, best_KF_score)
                self.best_result = result_dic
            
            return best_KF_score
      
        def save_best_model(self, model, score):
          self.best_score = score
          self.best_model = model
          now_time = datetime.datetime.now()
          time_stamp = str(now_time.year) + str(now_time.month) + str(now_time.day) + str(now_time.hour) + str(now_time.min)
          with open(self.directory_path + 'LGBM_OptunaBestModel_' + time_stamp + '.pickle', 'wb') as obj:
              pickle.dump(model, obj)