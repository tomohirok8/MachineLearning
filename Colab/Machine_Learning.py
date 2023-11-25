import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from matplotlib import rcParams
import lightgbm as lgb



### LightGBMで学習 ###
def Light_GBM(X_train, Y_train):
    rate_list = [0.05, 0.1, 0.2, 0.3, 0.4]
    depth_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    leaves_list = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    min_leaf_list = [1, 2, 3, 4, 5]

    # 学習回数
    esr = 300

    start_time = datetime.datetime.now()

    # トレーニングデータを学習用・検証用に分割
    X_trn, X_val, Y_trn, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

    # LightGBMの学習
    lgb_dataset_trn = lgb.Dataset(X_trn, label=Y_trn, categorical_feature='auto')
    lgb_dataset_val = lgb.Dataset(X_val, label=Y_val, categorical_feature='auto')

    # LightGBMのグリッドサーチ
    best_score = 0
    best_parameters = {}
    for rate in rate_list:
        for depth in depth_list:
            for leaves in leaves_list:
                for min_leaf in min_leaf_list:
                    params = {'objective' : 'multiclass',
                                'num_class' : 5, # 多クラスのクラス数を指定
                                # 'objective' : 'regression',
                                # 'objective' : 'binary', 
                                'metric': {'multi_error'},
                                # 'metric': {'mae'},
                                # 'metric': {'mse'},
                                # 'metric': {'binary_logloss'},
                                # 'metric': {'binary_error'}, # 評価指標 : 誤り率(= 1-正答率)
                                'early_stopping_rounds' : esr,   # early_stopping 回数指定
                                'learning_rate' : rate,
                                'max_depth' : depth,
                                'num_leaves': leaves,
                                'min_data_in_leaf': min_leaf
                                }

                    result_dic ={}
                    model = lgb.train(
                            params=params, 
                            train_set=lgb_dataset_trn, 
                            valid_sets=[lgb_dataset_trn, lgb_dataset_val], 
                            num_boost_round=10000, 
                            early_stopping_rounds=esr, 
                            # verbose_eval=100,
                            evals_result=result_dic
                            )

                    # train_pred = model.predict(X_trn, num_iteration=model.best_iteration)
                    train_pred_prob = model.predict(X_trn, num_iteration=model.best_iteration)
                    # train_pred = np.where(train_pred_prob < 0.5, 0, 1) # 0.5より小さい場合0 ,そうでない場合1を返す
                    train_pred = np.argmax(train_pred_prob, axis=1) # 最尤と判断したクラス
                    train_acc = accuracy_score(Y_trn.values, train_pred)
                    # val_pred = model.predict(X_val, num_iteration=model.best_iteration)
                    val_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
                    # val_pred = np.where(val_pred_prob < 0.5, 0, 1)
                    val_pred = np.argmax(val_pred_prob, axis=1) # 最尤と判断したクラス
                    val_acc = accuracy_score(Y_val.values, val_pred)
                    print("rate  = ", rate)
                    print("depth = ", depth)
                    print("leaves = ", leaves)
                    print("min_leaf = ", min_leaf)
                    print(f'train acc : {train_acc:.3f}%')
                    print(f'valid acc : {val_acc:.3f}%')
                    
                    # 最も良いスコアのパラメータとスコアを更新
                    score = val_acc
                    if score > best_score:
                        best_score = score
                        best_parameters = {'rate' : rate,
                                            'depth' : depth,
                                            'leaves' : leaves,
                                            'min_leaf' : min_leaf}

    print('Best score: {}'.format(best_score))
    print('Best parameters: {}'.format(best_parameters))

    # 最適パラメータのモデルで学習
    params = {'objective' : 'multiclass',
            'num_class' : 5, # 多クラスのクラス数を指定
            # 'objective' : 'regression',
            # 'objective' : 'binary', 
            'metric': {'multi_error'},
            # 'metric': {'mae'},
            # 'metric': {'mse'},
            # 'metric': {'binary_logloss'},
            # 'metric': {'binary_error'}, # 評価指標 : 誤り率(= 1-正答率)
            'early_stopping_rounds' : esr,   # early_stopping 回数指定
            'learning_rate' : best_parameters["rate"],
            'max_depth' : best_parameters["depth"],
            'num_leaves': best_parameters["leaves"],
            'min_data_in_leaf': best_parameters["min_leaf"]
            }

    result_dic ={}
    model = lgb.train(
            params=params, 
            train_set=lgb_dataset_trn, 
            valid_sets=[lgb_dataset_trn, lgb_dataset_val], 
            num_boost_round=10000, 
            early_stopping_rounds=esr, 
            # verbose_eval=100,
            evals_result=result_dic
            )

    # 学習経過を表示
    result_df = pd.DataFrame(result_dic['training']).add_prefix('train_').join(pd.DataFrame(result_dic['valid_1']).add_prefix('valid_'))
    fig, ax = plt.subplots(figsize=(11, 7))
    result_df[['train_multi_error', 'valid_multi_error']].plot(ax=ax)
    ax.set_ylabel('multi error')
    ax.set_xlabel('num of iteration')
    #ax.set_ylim(2, 8)
    ax.grid()

    # 予測
    Y_pred_prob = model.predict(X_train)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    df_lightGBM = pd.concat([X_train, Y_train], axis=1)
    df_lightGBM['予測'] = Y_pred


    # 特徴量の重要度出力
    feature_importance = pd.DataFrame({
                                        'feature_name' : model.feature_name(),
                                        'importance' : model.feature_importance(importance_type='gain'), 
                                        })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize = (16, 9))
    sns.set(font='Yu Gothic')
    sns.barplot(data=feature_importance, x='importance', y='feature_name')
    plt.savefig('feature_importance.png')

    calc_time = datetime.datetime.now() - start_time
    print(calc_time)

    return df_lightGBM, best_parameters, feature_importance






























