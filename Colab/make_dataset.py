import os
import glob
import pandas as pd
import numpy as np
import time
import datetime
from tqdm import tqdm
import json

from Scripts.Utility import sort_RATE_list, Label_target, File_target
from Scripts.make_dataset_RATE import make_classification_data
from Scripts.analysis import cal_unique, cal_outlier, multi_correlation, PCA_classification
from Scripts.plot import plot_box
from Scripts.plot import plot_slice_data
from Scripts.window import window_cut



####### データセット作成 #######
def make_dataset_classification(Preprocess_set, setting, RAM_names, project, data_set, Model_set):
    ### 前処理の前提
    # 波形長さの設定
    max_len = int(Preprocess_set.at['波形長さ', 1])
    
    # データの切り出しと間引き設定
    slice_st_time = float(Preprocess_set.at['データ切り出し開始時間', 1])
    slice_en_time = float(Preprocess_set.at['データ切り出し終了時間', 1])
    mabiki = int(Preprocess_set.at['間引き数', 1])
    
    # 標準化実行フラグ
    flag_std = int(Preprocess_set.at['標準化設定', 1])
    
    # 窓切り取り設定
    flag_window = int(Preprocess_set.at['窓切り取り設定', 1])
    window_num = int(Preprocess_set.at['窓の数', 1])
    window_size = int(Preprocess_set.at['窓の長さ', 1])
    if flag_window == 1:
        print('許容波形長さmin :', window_num + window_size)
        print('許容波形長さmax :', window_num * window_size)
    
    # 説明変数リスト
    df_RAMconv = pd.read_csv('process_data/学習用RAM名変換リスト_' + data_set + '.csv', header=0, index_col=0)
    explanatory_list = df_RAMconv.values.reshape(-1,).tolist()
    
    
    ### 学習用データ前処理
    # ファイル名と官能評点を読み込み
    df_file_RATE = pd.read_csv('process_data/ファイル名と官能評点_' + data_set + '.csv', header=None)
    print('学習用ファイル総数 : {}個'.format(len(df_file_RATE)))
    filenames_all = list(df_file_RATE[0])
    RATE_all = list(df_file_RATE[1])
    
    # 目的変数名
    target_name = setting.at['目的変数名', 1]
    # 結果保存フォルダ名
    savedir = 'process_data'
    
    
    # クラスとファイル名の変換辞書、逆変換辞書作成
    Defined_RATE = pd.read_csv('setting/官能評点リスト.csv', dtype=str, header=None).values.reshape(-1).tolist()
    drop_list = []
    for r in range(len(df_file_RATE)):
        if df_file_RATE.iat[r, 1] not in Defined_RATE:
            drop_list.append(df_file_RATE.index.to_numpy()[r])
    df_file_RATE.drop(drop_list, inplace=True)
    print('学習用ファイル総数（対象選定後） : {}個'.format(len(df_file_RATE)))
    print()
    
    filenames = list(df_file_RATE[0])
    RATE = list(df_file_RATE[1])
    RATE_list = sort_RATE_list(list(set(RATE)), Defined_RATE)
    RATE_ratio_list = []
    for r in RATE_list:
        RATE_ratio_list.append(RATE.count(r)/len(RATE))
        print('官能評点 {} の個数 : {} 個　（{:.2f}%）'.format(r, RATE.count(r), RATE.count(r)/len(RATE)*100))
    n_class = len(RATE_list)
    print('官能評点の種類 {}個'.format(n_class))
    print()
    
    class_mapping = Label_target(RATE_list)
    class_mapping_inverse = {v:k for k,v in class_mapping.items()}
    file_mapping = File_target(list(set(filenames)))
    file_mapping_inverse = {v:k for k,v in file_mapping.items()}
    f = open('process_data/class_mapping_' + data_set + '.json', 'w')
    json.dump(class_mapping, f)
    f.close()
    f = open('process_data/class_mapping_inverse_' + data_set + '.json', 'w')
    json.dump(class_mapping_inverse, f)
    f.close()
    
    # 時間RAM名
    if int(setting.at['RAM名秘匿化', 1]) == 1:
        RAM_time = 'TIME'
    else:
        RAM_time = 'ECUTIME'
    
    # 学習用データから不要な説明変数を削除
    remove_path = 'setting/remove_RAM.csv'
    remove_list = pd.read_csv(remove_path, header=None).values.reshape(-1).tolist()
    for n in remove_list:
        try:
            explanatory_list.remove(n)
        except:
            print('{}は元の説明変数リストに存在しません'.format(n))
    print('学習用説明変数RAM : ', explanatory_list)
    
    # 学習用説明変数を秘匿化解除して出力
    if int(setting.at['RAM名秘匿化', 1]) == 1:
        confident_cancel = []
        for e in explanatory_list:
            confident_cancel.append(df_RAMconv[df_RAMconv['秘匿化RAM名'] == e].index.to_numpy()[0])
        print('秘匿化解除後説明変数 : {}'.format(confident_cancel))
    print()
    
    
    ### 特徴量テーブルを作成
    I_df = make_classification_data(setting, RAM_names, project, data_set, Model_set,
                                    target_name, explanatory_list, filenames_all, RATE_all,
                                    class_mapping, RAM_time, slice_st_time, slice_en_time,
                                    mabiki, flag_std, flag_window, window_num, window_size, max_len)
    
    
    ### 対象選定を特徴量テーブルに反映
    subject_list = []
    for d in range(len(I_df)):
        for r in filenames:
            if r in I_df.at[d, 'ファイル名']:
                subject_list.append(I_df.index.to_numpy()[d])
                break
    I_df2 = I_df.filter(items=subject_list, axis=0)
    I_df2.reset_index(drop=True, inplace=True)
    

    ### テスト用に分離するファイル名リストを読み込んで学習用ファイルから除外する
    if int(setting.at['テスト用にファイルを分離', 1]) == 1:
        filepath = 'setting/' + Model_set.at['テスト用分離するファイル名リスト', 1]
        test_file_list = pd.read_csv(filepath, header=0).values.reshape(-1).tolist()
        test_list = []
        for d in range(len(I_df2)):
            for r in test_file_list:
                if r in I_df2.at[d, 'ファイル名']:
                    test_list.append(I_df2.index.to_numpy()[d])
                    break
        I_df3 = I_df2.filter(items=test_list, axis=0)
        I_df3.sort_values(by='ファイル名', inplace=True)
        I_df3.reset_index(drop=True, inplace=True)
        I_df2.drop(test_list, inplace=True)
        I_df2.sort_values(by='ファイル名', inplace=True)
        I_df2.reset_index(drop=True, inplace=True)
        print('学習用ファイル総数（テスト用分離後） : {}個'.format(len(I_df2)))
        RATE = I_df2[target_name].values.tolist()
        for r in RATE_list:
            RATE_ratio_list.append(RATE.count(r)/len(RATE))
            print(' -官能評点 {} の個数 : {} 個　（{:.2f}%）'.format(r, RATE.count(r), RATE.count(r)/len(RATE)*100))
        print()
        
        print('テスト用ファイル総数 : {}個'.format(len(I_df3)))
        RATE_test = I_df3[target_name].values.tolist()
        for r in RATE_list:
            RATE_ratio_list.append(RATE_test.count(r)/len(RATE_test))
            print(' -官能評点 {} の個数 : {} 個　（{:.2f}%）'.format(r, RATE_test.count(r), RATE_test.count(r)/len(RATE_test)*100))
        print()
    
    
    ####### ユニーク値の分布を計算する #######
    print('【前処理】 ユニーク値の分布を計算')
    I_df2 = cal_unique(I_df2, n_class, RATE_ratio_list, target_name, savedir)
    explanatory_list_tf = list(I_df2.columns)
    print(' -学習に使う説明変数（ユニーク値ドロップ後）: {}個'.format(len(explanatory_list_tf)-2))
    print()
    
    
    ####### 各特徴量の外れ値を目的変数のクラスごとに計算 #######
    if int(setting.at['【前処理】特徴量の外れ値割合を計算', 1]) == 1:
        print('【前処理】 特徴量とファイルについて目的変数ごとに外れ値を計算')
        remove_feat_outlist, remove_feat_far_outlist\
            = cal_outlier(RATE_list, I_df2, target_name, explanatory_list_tf, savedir, setting, Preprocess_set, data_set)
        print()

    
    ####### 目的変数を数値化 #######
    I_df2[target_name] = I_df2[target_name].map(class_mapping)
    if int(setting.at['テスト用にファイルを分離', 1]) == 1:
        I_df3[target_name] = I_df3[target_name].map(class_mapping)
    
    
    ####### 学習用の特徴量リスト作成 #######
    print('【前処理】 特徴量の除外リストから読み込んでドロップ')
    remove_path_tf = 'setting/remove_Feature.csv'
    try:
        remove_list_tf = pd.read_csv(remove_path_tf, header=None).values.reshape(-1).tolist()
        for n in remove_list_tf:
            try:
                explanatory_list_tf.remove(n)
            except:
                pass
    except:
        pass
    explanatory_list_tf.remove('ファイル名')
    explanatory_list_tf.remove(target_name)
    print(' -学習に使う説明変数（除外リスト適用後）: {}個'.format(len(explanatory_list_tf)))
    print()
    
    
    ####### 特徴量をプロット #######
    print('【可視化】 特徴量の分布を箱ひげ図にプロット')
    os.makedirs('process_data/特徴量分布_学習用', exist_ok=True)
    for files in os.scandir('process_data/特徴量分布_学習用'):
        os.remove(files.path)
    plot_box(explanatory_list_tf, target_name, I_df2, class_mapping_inverse, '特徴量分布_学習用')
    if int(setting.at['テスト用にファイルを分離', 1]) == 1:
        os.makedirs('process_data/特徴量分布_テスト用', exist_ok=True)
        for files in os.scandir('process_data/特徴量分布_テスト用'):
            os.remove(files.path)
        plot_box(explanatory_list_tf, target_name, I_df3, class_mapping_inverse, '特徴量分布_テスト用')
    if int(setting.at['【前処理】特徴量の外れ値割合を計算', 1]) == 1:
        os.makedirs('process_data/外れ値割合が高い特徴量の分布', exist_ok=True)
        os.makedirs('process_data/外れ度が高い特徴量の分布', exist_ok=True)
        for files in os.scandir('process_data/外れ値割合が高い特徴量の分布'):
            os.remove(files.path)
        for files in os.scandir('process_data/外れ度が高い特徴量の分布'):
            os.remove(files.path)
        plot_box(remove_feat_outlist, target_name, I_df2, class_mapping_inverse, '外れ値割合が高い特徴量の分布')
        plot_box(remove_feat_far_outlist, target_name, I_df2, class_mapping_inverse, '外れ度が高い特徴量の分布')
    print()
    
    
    ####### 多重共線性 #######
    print('【前処理】 多重共線性チェック')
    multi_correlation(I_df2, explanatory_list_tf, explanatory_list, target_name, savedir, setting, Preprocess_set, data_set, '【学習】')
    if int(setting.at['テスト用にファイルを分離', 1]) == 1:
        multi_correlation(I_df3, explanatory_list_tf, explanatory_list, target_name, savedir, setting, Preprocess_set, data_set, '【テスト】')
    print()
    
    
    ####### 行列のランクを計算 #######
    # n次正方行列Aについて rank(A)<n となることをランク落ちという
    rank_df2 = np.linalg.matrix_rank(I_df2[explanatory_list_tf].values)
    shape_df2 = I_df2[explanatory_list_tf].values.shape
    print('学習に使う説明変数行列の形状: {} × {}'.format(shape_df2[0], shape_df2[1]))
    print('学習に使う説明変数行列のランク: {}'.format(rank_df2))

    
    ####### 主成分分析 #######
    if int(setting.at['【前処理】特徴量の主成分分析', 1]) == 1:
        os.makedirs('process_data/', exist_ok=True)
        PCA_classification(I_df2, target_name, explanatory_list_tf, class_mapping_inverse)
    
    
    return I_df2, explanatory_list_tf, target_name, RATE_list, n_class, class_mapping, class_mapping_inverse





def make_dataset_regression_TimeSeries(Preprocess_set, setting, RAM_names, project, data_set, Model_set):
    os.makedirs('process_data/slice_data_plot_' + data_set, exist_ok=True)
    for files in os.scandir('process_data/slice_data_plot_' + data_set):
        os.remove(files.path)
        
    ### 前処理の前提
    # 波形長さの設定
    max_len = int(Preprocess_set.at['波形長さ', 1])
    
    # データの切り出しと間引き設定
    slice_st_time = float(Preprocess_set.at['データ切り出し開始時間', 1])
    slice_en_time = float(Preprocess_set.at['データ切り出し終了時間', 1])
    mabiki = int(Preprocess_set.at['間引き数', 1])
    
    # 標準化実行フラグ
    flag_std = int(Preprocess_set.at['標準化設定', 1])
    
    # 窓切り取り設定
    flag_window = int(Preprocess_set.at['窓切り取り設定', 1])
    window_num = int(Preprocess_set.at['窓の数', 1])
    window_size = int(Preprocess_set.at['窓の長さ', 1])
    if flag_window == 1:
        print('許容波形長さmin :', window_num + window_size)
        print('許容波形長さmax :', window_num * window_size)
    
    # 説明変数リスト
    df_RAMconv = pd.read_csv('process_data/学習用RAM名変換リスト_' + data_set + '.csv', header=0, index_col=0)
    explanatory_list = df_RAMconv.values.reshape(-1,).tolist()
    
    
    ### 学習用データ前処理
    # ファイル名を読み込み
    pattern_list = glob.glob('process_data/train_' + data_set + '/*.csv')
    # pattern_list = pattern_list[:100]
    print('学習用ファイル総数 : {}個'.format(len(pattern_list)))
    print()
    
    # 目的変数名
    target_name = setting.at['目的変数名', 1]
    # 結果保存フォルダ名
    savedir = 'process_data'
    
    # 各説明変数、目的変数ごとにの最大値を算出
    df_RAM_max_min = 'none'
    if flag_std == 2:
        print('### 最大値をスペックから直接指定してcsv出力 ###')
        tmp_all = []
        for i in range(len(pattern_list)):
            tmp_data = pd.read_csv(pattern_list[i], header=0)[::mabiki]
            tmp_all.append(tmp_data)
        df_all = pd.concat(tmp_all, axis=0)
        
        col_all = list(df_all.columns)
        RAM_max_min = []
        for c in col_all:
            RAM_max_min.append([c, df_all[c].max(), df_all[c].min()])
        
        df_RAM_max_min = pd.DataFrame(RAM_max_min, columns=['RAM', 'max', 'min'])
        # for i in range(len(df_RAM_max_min)):
            # if df_RAM_max_min.at[i, 'RAM'] == 'RAM7':
            #     df_RAM_max_min.at[i, 'max'] = 8000.0 # データの最大7514
            #     df_RAM_max_min.at[i, 'min'] = 0.0
            # elif df_RAM_max_min.at[i, 'RAM'] == 'RAM_T':
            #     df_RAM_max_min.at[i, 'max'] = 3500.0 # データの最大3344
            #     df_RAM_max_min.at[i, 'min'] = -50.0 # データの最小-44
            # elif df_RAM_max_min.at[i, 'RAM'] == 'RAM41':
            #     df_RAM_max_min.at[i, 'max'] = 900.0 # データの最大899
            #     df_RAM_max_min.at[i, 'min'] = -50.0 # データの最小-47
            # elif df_RAM_max_min.at[i, 'RAM'] == 'RAM34':
            #     df_RAM_max_min.at[i, 'max'] = 1.0 # データの最大0.99
            #     df_RAM_max_min.at[i, 'min'] = -0.2 # データの最小-0.14
            # elif df_RAM_max_min.at[i, 'RAM'] == 'RAM42':
            #     df_RAM_max_min.at[i, 'max'] = 1.0
            #     df_RAM_max_min.at[i, 'min'] = -0.2
        df_RAM_max_min.to_csv('process_data/各RAMの最大値と最小値_' +  data_set + '.csv', index=False, encoding='utf_8_sig')
    
    # ファイル名辞書作成
    pattern = []
    for f in pattern_list:
        pattern.append(os.path.basename(f).replace('.csv',''))
    file_mapping = File_target(list(set(pattern)))
    file_mapping_inverse = {v:k for k,v in file_mapping.items()}
    
    # 時間RAM名
    if int(setting.at['RAM名秘匿化', 1]) == 1:
        RAM_time = 'TIME'
        RAM_xsft = 'RAM_X1'
    else:
        RAM_time = 'ECUTIME'
        RAM_xsft = 'xsft'
    
    
    # 学習用データから不要な説明変数を削除
    remove_path = 'setting/remove_RAM.csv'
    remove_list = pd.read_csv(remove_path, header=None).values.reshape(-1).tolist()
    for n in remove_list:
        try:
            explanatory_list.remove(n)
        except:
            print('{}は元の説明変数リストに存在しません'.format(n))
    print('学習用説明変数RAM : ', explanatory_list)
    
    # 学習用説明変数を秘匿化解除して出力
    if int(setting.at['RAM名秘匿化', 1]) == 1:
        confident_cancel = []
        for e in explanatory_list:
            confident_cancel.append(df_RAMconv[df_RAMconv['秘匿化RAM名'] == e].index.to_numpy()[0])
        print('秘匿化解除後説明変数 : {}'.format(confident_cancel))
    print()
    
    ### テスト用に分離するファイル名リストを読み込んで学習用ファイルから除外する
    if int(setting.at['テスト用にファイルを分離', 1]) == 1:
        filepath = 'setting/' + Model_set.at['テスト用分離するファイル名リスト', 1]
        test_file_list = pd.read_csv(filepath, header=0).values.reshape(-1).tolist()
        test_list = []
        for p in pattern_list:
            for r in test_file_list:
                if r in p:
                    test_list.append(p)
                    break
        pattern_list = list(set(pattern_list) - set(test_list))
        print('学習用ファイル総数（テスト用分離後） : {}個'.format(len(pattern_list)))
        print()
    
    
    t_start = time.time()
    
    X = []
    Y = []
    len_list = []
    max_list = [0] * len(explanatory_list)
    for i in tqdm(range(len(pattern_list))):
        filename = os.path.basename(pattern_list[i]).replace('.csv','')
        file_No = file_mapping[filename]
    
        # 波形部分を読み込み
        tmp_data = pd.read_csv(pattern_list[i], header=0)[::mabiki].reset_index(drop=True)
        tmp_data[RAM_time] = tmp_data[RAM_time] - tmp_data.loc[0,RAM_time]
        
        # xsft=1のところだけ切り出し
        xsft = tmp_data[RAM_xsft]
        slice_st = 0
        slice_en = len(xsft)-1
        for k in range(1,len(xsft)):
            if xsft[k] == 1 and xsft[k-1] == 0:
                slice_st = k
            if xsft[k] == 0 and xsft[k-1] == 1 and slice_st > 0:
                slice_en = k
                break
        if slice_st == 0:
            print('変速部分の切り出し失敗')
        
        # 時系列データ設定
        sample_time = round(tmp_data.loc[1,RAM_time] - tmp_data.loc[0,RAM_time],4)
        
        # 切り出し波形サイズ
        slice_st = min(slice_st, int(slice_st_time / sample_time))
        slice_en = max(slice_en, int(slice_en_time / sample_time))
        slice_len = slice_en - slice_st
        len_list.append(slice_len)

        # 説明変数の最大値
        for k, name in enumerate(explanatory_list):
            name_data = tmp_data[name].values[slice_st:slice_en]
            for n in name_data:
                if n > max_list[k]:
                    max_list[k] = n
        
        # 説明変数の標準化処理
        name_data_std = []
        for name in explanatory_list:
            name_data = tmp_data[name].values[slice_st:slice_en]
            if flag_std == 1:
                ram_max = np.max(name_data)
                ram_min = np.min(name_data)
                tmp_std = (name_data - ram_min) / (ram_max - ram_min)
                name_data_std.append(tmp_std)
            elif flag_std == 2:
                for j in range(len(df_RAM_max_min)):
                    if name == df_RAM_max_min.at[j,'RAM']:
                        ram_max = float(df_RAM_max_min.at[j,'max'])
                        ram_min = float(df_RAM_max_min.at[j,'min'])
                        tmp_std = (name_data - ram_min) / (ram_max - ram_min)
                        name_data_std.append(tmp_std)
                        break
            else:
                name_data_std.append(name_data)
    
        # 説明変数データを窓カットしてXに追加する準備（窓カットしない場合はパディング）
        RAM_train = []
        for name, ram_data in zip(explanatory_list, name_data_std):
            if flag_window == 1:
                name_data_add = window_cut(ram_data, window_num, window_size)
                for add in name_data_add:
                    RAM_train.append(add)
            else:
                # スライスorパディング
                if slice_len < max_len:
                    pad_length = max_len - slice_len
                    RAM_pad = np.zeros((1, pad_length)).reshape(-1)
                    RAM_train.append(np.hstack([ram_data, RAM_pad]))
                elif slice_len > max_len:
                    RAM_train.append(ram_data[:max_len])
                else:
                    RAM_train.append(ram_data)
            
        RAM_train = np.array(RAM_train).T
        
        # スライス後波形の確認
        if flag_window != 1:
            plot_slice_data(tmp_data, slice_st, slice_en, filename, RAM_names, project, data_set)
            
        ### 目的変数
        # 標準化処理
        target_data = tmp_data[target_name].values[slice_st:slice_en]
        if flag_std == 1:
            ram_max = np.max(target_data)
            ram_min = np.min(target_data)
            target_data_std = (target_data - ram_min) / (ram_max - ram_min)
        elif flag_std == 2:
            # print('### 最大値をスペックから直接指定して標準化 ###')
            for j in range(len(df_RAM_max_min)):
                if target_name == df_RAM_max_min.at[j,'RAM']:
                    ram_max = float(df_RAM_max_min.at[j,'max'])
                    ram_min = float(df_RAM_max_min.at[j,'min'])
                    tmp_std = (name_data - ram_min) / (ram_max - ram_min)
                    target_data_std = (target_data - ram_min) / (ram_max - ram_min)
                    break
        else:
            target_data_std = target_data
            ram_max = 1
            ram_min = -1
        
        # スライスorパディング
        if slice_len < max_len:
            pad_length = max_len - slice_len
            RAM_pad = np.zeros((1, pad_length)).reshape(-1)
            RAM_target = np.hstack([target_data_std, RAM_pad])
        elif slice_len > max_len:
            RAM_target = target_data_std[:max_len]
        else:
            RAM_target = target_data_std
        
        file_No_list = np.array([file_No] * len(RAM_target))
        ram_max_list = np.array([ram_max] * len(RAM_target))
        ram_min_list = np.array([ram_min] * len(RAM_target))
        
        tmp_Y = np.vstack([RAM_target, file_No_list, ram_max_list, ram_min_list]).T
        
        # リストに追加
        X.append(RAM_train)
        Y.append(tmp_Y)
    
    t_end = time.time()
    print(datetime.timedelta(seconds = round(t_end - t_start,0)))

    return X, Y, sample_time, len_list



