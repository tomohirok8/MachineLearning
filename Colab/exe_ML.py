import os
import pandas as pd

from Scripts.Utility import cal_acc, make_mixed_matrix, timedelta_to_hms
from Scripts.plot import plot_significant_feature, plot_2axis
from Scripts.MachineLearning import Light_GBM, Light_GBM_Optuna



def ML(I_df, explanatory_list, target_name, RATE_list, n_class, class_mapping, class_mapping_inverse, setting, Model_set, data_set):
    ### 結果保存フォルダ
    savedir = 'result'
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir + '/寄与度の高い特徴量分布', exist_ok=True)
    for files in os.scandir(savedir + '/寄与度の高い特徴量分布'):
        os.remove(files.path)
    os.makedirs(savedir + '/寄与度上位の特徴量2軸プロット', exist_ok=True)
    for files in os.scandir(savedir + '/寄与度上位の特徴量2軸プロット'):
        os.remove(files.path)
        
    ####### 学習実行 #######
    print()
    print('#'*14, ' 学習開始 ', '#'*14)
    print()
    
    ### LightGBMを実行
    if int(setting.at['【学習】LightGBMで分類', 1]) == 1:
        exe_LightGBM(I_df, explanatory_list, target_name, RATE_list, n_class, class_mapping, class_mapping_inverse,
                     setting, Model_set, data_set, savedir)
    else:
        print('学習を実行しません->処理を終了します')

    



### LightGBM ###
def exe_LightGBM(I_df, explanatory_list, target_name, RATE_list, n_class, class_mapping, class_mapping_inverse,
                 setting, Model_set, data_set, savedir):
    X = I_df[explanatory_list]
    Y = I_df[target_name]
    file_name = I_df['ファイル名']
        
    if int(setting.at['【学習】Optunaでハイパーパラメータ最適化', 1]) == 1:
        df_LGBM, feature_importance, calc_time\
            = Light_GBM_Optuna(X, Y, file_name, n_class, class_mapping, savedir)
    else:
        HP_GridSearch_list = pd.read_csv('setting/' + Model_set.at['ハイパーパラメータのグリッドサーチリスト', 1], header=0)
        df_LGBM, feature_importance, calc_time\
            = Light_GBM(X, Y, file_name, n_class, class_mapping, HP_GridSearch_list, savedir)
    
    y_tgt = df_LGBM[target_name]
    y_pred = df_LGBM['予測']
    accuracy, recall, precision, f1 = cal_acc(y_pred, y_tgt, class_mapping)
    print('正解率 : {:.3f}%'.format(accuracy))
    print('再現率 : {:.3f}%'.format(recall))
    print('適合率 : {:.3f}%'.format(precision))
    print('F1    : {:.3f}%'.format(f1))
    
    # 予測結果のラベルを目標変数の表現に戻す
    target_list = []
    for v in df_LGBM[target_name].values:
        target_list.append(class_mapping_inverse[v])
    pred_list = []
    for v in df_LGBM['予測'].values:
        pred_list.append(class_mapping_inverse[v])
    df_LGBM.drop(columns=[target_name, '予測'], inplace=True)
    df_LGBM.insert(loc=1, column=target_name, value=target_list)
    df_LGBM.insert(loc=2, column='予測', value=pred_list)

    # 混同行列を作る
    matrix = make_mixed_matrix(df_LGBM, target_name, RATE_list)
    
    # csv保存
    df_LGBM.to_csv(savedir + '/【LightGBM】LightGBM結果.csv', index = False, encoding='utf_8_sig')
    feature_importance.to_csv(savedir + '/【LightGBM】各特徴量の寄与度.csv', index = False, encoding='utf_8_sig')
    matrix.to_csv(savedir + '/【LightGBM】LightGBM結果の混同行列.csv', encoding='utf_8_sig')
    
    # 寄与度の高い特徴量をプロット
    plot_significant_feature(feature_importance, target_name, I_df, class_mapping_inverse, savedir)
    
    # 寄与度上位の特徴量を2軸プロット
    plot_2axis(feature_importance, target_name, df_LGBM, RATE_list, savedir)
    
    h, m, s = timedelta_to_hms(calc_time)
    print('LightGBM実行時間 : {:02d}hr {:02d}min {:.1f}sec'.format(h, m, s))

