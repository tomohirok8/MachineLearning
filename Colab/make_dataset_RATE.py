import os
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import from_columns
from tqdm import tqdm
import copy
import json

from Scripts.plot import plot_slice_data
from Scripts.window import window_cut




def make_classification_data(setting, RAM_names, project, data_set, Model_set,
                             target_name, explanatory_list, filenames_all, RATE_all,
                             class_mapping, RAM_time, slice_st_time, slice_en_time,
                             mabiki, flag_std, flag_window, window_num, window_size, max_len):
    
    ### ドメイン知識の特徴量を読み込み
    if int(setting.at['【前処理】ドメイン知識特徴量作成', 1]) == 1:
        filepath = 'process_data/' + Model_set.at['ドメイン知識特徴量リスト', 1]
        I_df_domain = pd.read_csv(filepath, header=0)
        print('学習に使う説明変数（ドメイン知識で作成した特徴量）: {}個'.format(len(list(I_df_domain.columns))-2))
    
    
    ### tsfreshで特徴量を作成する場合
    if int(setting.at['【前処理】tsfresh特徴量作成', 1]) == 1 and int(setting.at['【前処理】tsfresh作成済特徴量を使用', 1]) != 1:
        I_df_tsfresh\
            = make_dataset_tsfresh(target_name, explanatory_list, filenames_all, RATE_all, class_mapping,
                                   mabiki, RAM_time, flag_std, flag_window,
                                   window_num, window_size, max_len, setting, Model_set,
                                   slice_st_time, slice_en_time, RAM_names, project, data_set)
        print('学習に使う説明変数（tsfreshが生成した特徴量）: {}個'.format(len(list(I_df_tsfresh.columns))-2))
    
    
    ### tsfreshで作成した特徴量を読み込む場合
    elif int(setting.at['【前処理】tsfresh作成済特徴量を使用', 1]) == 1:
        I_df_tsfresh = use_dataset_tsfresh(target_name, data_set, Model_set)
        print('学習に使う説明変数（tsfreshが生成した特徴量を読み込み）: {}個'.format(len(list(I_df_tsfresh.columns))-2))
    
    
    ### tsfresh特徴量の指定リスト使用
    if int(setting.at['【前処理】tsfresh特徴量の指定リストを適用', 1]) == 1:
        filepath = 'setting/' + Model_set.at['tsfresh特徴量の指定リスト', 1]
        tsfresh_pickup = pd.read_csv(filepath, header=None).values.reshape(-1).tolist()
        I_df_tsfresh = I_df_tsfresh.filter(items=['ファイル名', target_name] + tsfresh_pickup, axis=1)
        I_df_tsfresh.sort_values(by='ファイル名', inplace=True)
        I_df_tsfresh.reset_index(drop=True, inplace=True)
        print('学習に使う説明変数（tsfresh特徴量の指定リスト適用後）: {}個'.format(len(list(I_df_tsfresh.columns))-2))
    
    
    ### 特徴量テーブルの結合
    if (int(setting.at['【前処理】tsfresh特徴量作成', 1]) == 1 or int(setting.at['【前処理】tsfresh作成済特徴量を使用', 1]) == 1)\
        and int(setting.at['【前処理】ドメイン知識特徴量作成', 1]) == 1:
            for i in range(len(I_df_domain)):
                if I_df_domain.iat[i,0] != I_df_tsfresh.iat[i,0]:
                    print('ファイル名ソート失敗')
                    print('ドメイン知識 : {} <-> tsfresh : {}'.format(I_df_domain.iat[i,0], I_df_tsfresh.iat[i,0]))
            I_df_tsfresh.drop(['ファイル名', '官能評点'], axis=1, inplace=True)
            I_df2 = pd.concat([I_df_domain, I_df_tsfresh], axis=1)
    elif (int(setting.at['【前処理】tsfresh特徴量作成', 1]) == 1 or int(setting.at['【前処理】tsfresh作成済特徴量を使用', 1]) == 1)\
        and int(setting.at['【前処理】ドメイン知識特徴量作成', 1]) != 1:
            I_df2 = I_df_tsfresh.copy()
    elif (int(setting.at['【前処理】tsfresh特徴量作成', 1]) != 1 and int(setting.at['【前処理】tsfresh作成済特徴量を使用', 1]) != 1)\
        and int(setting.at['【前処理】ドメイン知識特徴量作成', 1]) == 1:
            I_df2 = I_df_domain.copy()
    else:
        I_df2 = None
        print('特徴量テーブルを読み込みません')
        
    print()
    return I_df2





####### tsfreshによる特徴量作成 #######
def make_dataset_tsfresh(target_name, explanatory_list, pattern, RATE, class_mapping,
                         mabiki, RAM_time, flag_std, flag_window,
                         window_num, window_size, max_len, setting, Model_set,
                         slice_st_time, slice_en_time, RAM_names, project, data_set):
    os.makedirs('process_data/slice_data_plot_' + data_set, exist_ok=True)
    for files in os.scandir('process_data/slice_data_plot_' + data_set):
        os.remove(files.path)
    # os.makedirs('process_data/autofeat', exist_ok=True)
    # for files in os.scandir('process_data/autofeat' + data_set):
    #     os.remove(files.path)
    
    I_list = []
    rate_list = []
    slice_len_list = []
    cut_data_ID = 0
    for i in tqdm(range(len(pattern))):
        pattern_file = 'process_data/train_' + data_set + '/' + pattern[i] + '.csv'
        filename = pattern_file.split('/')[-1].replace('.csv','')
        cut_data_ID += 1            
        # 波形部分を読み込み
        tmp_data = pd.read_csv(pattern_file, header=0)[::mabiki].reset_index(drop=True)
        tmp_data[RAM_time] = tmp_data[RAM_time] - tmp_data.loc[0,RAM_time]
        
        # 時系列データ設定
        sample_time = round(tmp_data.loc[1,RAM_time] - tmp_data.loc[0,RAM_time],4)
        if data_set == '070D_MONUP' or data_set == '070D_MOFFDWN':
            slice_st = int(slice_st_time / sample_time)
            slice_en = len(tmp_data) - int(slice_en_time / sample_time)
        slice_len = slice_en - slice_st
        slice_len_list.append(slice_len)
        # print('tsfresh用切り出し波形長さ', slice_len)
        
        # スライス後波形の確認
        if data_set == '070D_MONUP' or data_set == '070D_MOFFDWN':
            plot_slice_data(tmp_data, slice_st, slice_en, filename, RAM_names, project, data_set)
                
        # 説明変数の標準化処理
        name_data_std = []
        for name in explanatory_list:
            name_data = tmp_data[name].values[slice_st:slice_en]
            if flag_std == 1:
                ram_max = np.max(name_data)
                ram_min = np.min(name_data)
                tmp_std = (name_data - ram_min) / (ram_max - ram_min)
                name_data_std.append(tmp_std)
            else:
                name_data_std.append(name_data)
        
        # 説明変数データを窓カットしてXに追加する準備
        RAM_train = []
        for name, ram_data in zip(explanatory_list, name_data_std):
            # パディングする
            pad_length = max_len - slice_len
            RAM_pad = np.zeros((1, pad_length)).reshape(-1)
            ram_data_pad = np.hstack([ram_data, RAM_pad])
            # リストに追加
            if flag_window == 1:
                name_data_add = window_cut(ram_data_pad, window_num, window_size)
                for add in name_data_add:
                    RAM_train.append(add)
            else:
                RAM_train.append(ram_data_pad)
        
        if flag_window == 1:
            RAM_train.append(tmp_data[RAM_time].values[:window_size])
        else:
            RAM_train.append(tmp_data[RAM_time].values[:slice_len])
    
        RAM_train = np.array(RAM_train).T
        
        # 窓切り出し説明変数名リスト作成
        extend_explanatory_list = []
        if flag_window == 1:
            for e in explanatory_list:
                for n in range(window_num):
                    extend_explanatory_list.append(e + '_' + str(n+1))
        else:
            extend_explanatory_list = copy.copy(explanatory_list)
        
        df_cut_comb = pd.DataFrame(RAM_train, columns=['time'] + extend_explanatory_list)
        # df_cut_comb.to_csv('process_data/autofeat/切り出し波形_' + filename + '.csv', index=False, encoding='utf_8_sig')
        
        if flag_window == 1:
            df_cut_comb.insert(loc=0, column='ID', value=[str(cut_data_ID)] * window_size)
        else:
            df_cut_comb.insert(loc=0, column='ID', value=[str(cut_data_ID)] * slice_len)
        
        # 特徴量をリストに追加
        I_list.append(df_cut_comb)
        rate_list.append([filename, str(cut_data_ID), RATE[i]])
        
    # 特徴量リストのファイル出力時接尾辞
    if flag_window == 1:
        name_tail = '_v' + str(len(explanatory_list)) + 'w' + str(window_num)
    else:
        name_tail = '_v' + str(len(explanatory_list)) + 'w1'
    
    
    I_df = pd.concat(I_list)
    df_rate = pd.DataFrame(rate_list, columns=['ファイル名', 'ID', '官能評点'])
    
    extracted_features = extract_features(I_df,
                                          column_id="ID",
                                          column_sort="time")
    
    settings = from_columns(extracted_features)
    f = open('process_data/tsfresh_settings.json', 'w')
    json.dump(settings, f)
    f.close()
    
    extracted_features.reset_index(inplace=True)
    extracted_features['index'] = extracted_features['index'].astype('int')
    extracted_features.sort_values(by='index', inplace=True)
    extracted_features.reset_index(drop=True, inplace=True)
    extracted_features.drop('index', axis=1, inplace=True)
    
    # 自動生成特徴量名を通しNoに変換
    AF_columns = list(extracted_features.columns)
    feat_list = []
    for feat in AF_columns:
        if '"' in feat:
            feat = feat.replace('"','')
        if ',' in feat:
            feat = feat.replace(',','_')
        feat_list.append(feat)
    
    extracted_features.set_axis(feat_list, axis='columns', inplace=True)
    
    # 自動生成した特徴量を選別、編集
    impute(extracted_features)
    
    # 目的変数列を結合
    I_df2 = pd.concat([df_rate, extracted_features], axis=1)
    I_df2.drop('ID', axis=1, inplace=True)
    I_df2.sort_values(by='ファイル名', inplace=True)
    
    # tsfreshで作成した特徴量リストを出力（除外するものを選定するかどうかでファイル名を変える）
    if int(setting.at['【前処理】tsfresh特徴量の除外リストを適用', 1]) == 1:
        # tsfreshの特徴量から除外するものリストを読み込む
        filepath = 'setting/' + Model_set.at['tsfresh特徴量の除外リスト', 1]
        tsfresh_remove = pd.read_csv(filepath, header=None).values.reshape(-1).tolist()
    
        # tsfreshの特徴量から除外    
        drop_list = []
        for f in feat_list:
            for r in tsfresh_remove:
                if r in f:
                    drop_list.append(f)
                
        ts_columns = []
        for c in feat_list:
            if c not in drop_list:
                ts_columns.append(c)
            
        I_df2_mod = I_df2[['ファイル名', target_name] + ts_columns]
        I_df2_mod.reset_index(drop=True, inplace=True)
        
        # csv出力
        I_df2_mod.to_csv('process_data/tsfresh特徴量テーブル_除外あり'\
                        + str(len(I_df2_mod.columns)-2) + name_tail + '_' + data_set + '.csv', index=False, encoding='utf_8_sig')
    
    else:
        I_df2_mod = I_df2
        I_df2_mod.reset_index(drop=True, inplace=True)
        # csv出力
        I_df2.to_csv('process_data/tsfresh特徴量テーブル_'\
                        + str(len(I_df2.columns)-2) + name_tail + '_' + data_set + '.csv', index=False, encoding='utf_8_sig')
    
    if flag_window == 1:
        print('波形の長さ(最小) :　{}個 {}sec'.format(min(slice_len_list), round(min(slice_len_list) * sample_time, 3)))
        print('波形の長さ(最大) : {}個 {}sec'.format(max(slice_len_list), round(max(slice_len_list) * sample_time, 3)))
        print('窓の長さ : {}個 {}sec'.format(window_size, round(window_size * sample_time, 3)))
        print('窓の個数 : {}個'.format(window_num))
    else:
        print('波形の長さ[sec] :', round(slice_len * sample_time, 3))

    return I_df2_mod





####### tsfreshによる作成済特徴量を使う #######
def use_dataset_tsfresh(target_name, data_set, Model_set):
    # 作成済特徴量ファイルを読み込み
    filepath = 'process_data/' + Model_set.at['tsfresh作成済特徴量リスト', 1]
    I_df2 = pd.read_csv(filepath, header=0)
    I_df2.sort_values(by='ファイル名', inplace=True)
    I_df2.reset_index(drop=True, inplace=True)
            
    return I_df2

