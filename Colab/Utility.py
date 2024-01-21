import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from scipy import fftpack
from scipy import signal
import pandas as pd
import random



# timedelta形式の時間を書式変換
def timedelta_to_hms(td):
    sec = td.total_seconds()
    h = int(sec//3600)
    m = int(sec%3600//60)
    s = sec%3600%60
    return h, m, s





# FFTフィルタをかける
def FFT_filter(wave, sample_time, FFT_filter):
    # FFTでGにフィルタ
    sample_freq = fftpack.fftfreq(len(wave),d=sample_time)
    sig_fft = fftpack.fft(wave)
    
    sig_fft[np.abs(sample_freq) > FFT_filter] = 0
    wave_FFT =np.real(fftpack.ifft(sig_fft))
        
    return wave_FFT



# バターワースフィルタをかける
def Butter_filter(wave, sample_time, cutoff_frequency, order):
    fs = 1 / sample_time
    fn = 0.5 * fs
    low = cutoff_frequency / fn
    b, a = signal.butter(order, low, btype='low')
    wave_Butter = signal.filtfilt(b, a, wave)
    
    return wave_Butter



def sort_RATE_list(RATE_list, Defined_RATE):
    # 官能評点を手動並び替え
    RATE_list_mod = []
    for d in Defined_RATE:
        if d in RATE_list:
            RATE_list_mod.append(d)
    
    return RATE_list_mod
    


def Label_target(label_list):
    # 各カテゴリごとに整数値を割り当てる
    class_mapping = {}
    for i in range(len(label_list)):
        class_mapping[label_list[i]] = i
    
    return class_mapping



def File_target(label_list):
    # 各カテゴリごとに整数値を割り当てる
    class_mapping = {}
    for i in range(len(label_list)):
        class_mapping[label_list[i]] = i
    
    return class_mapping





# 多クラス分類の精度を計算
def cal_acc(pred, tgt, class_mapping):
    target_list = list(class_mapping.values())

    TP_list = []
    tgt_sum = []
    pred_sum = []
    for n in target_list:
        TP = 0
        for t, p in zip(tgt, pred):
            if n == t and n == p:
                TP += 1
        TP_list.append(TP)
        tgt_sum.append(tgt.tolist().count(n))
        pred_sum.append(pred.tolist().count(n))

    recall_list = []
    for tp, t in zip(TP_list, tgt_sum):
        if t == 0:
            recall_list.append(0)
        else:
            recall_list.append(tp / t * 100)

    precision_list = []
    for tp, p in zip(TP_list, pred_sum):
        if p == 0:
            precision_list.append(0)
        else:
            precision_list.append(tp / p * 100)

    fi_list = []
    for r, p in zip(recall_list, precision_list):
        if r + p == 0:
            fi_list.append(0)
        else:
            fi_list.append(2 * r * p /(r + p))
    
    recall = round(np.mean(np.array(recall_list)), 3)
    precision = round(np.mean(np.array(precision_list)), 3)
    f1 = round(np.mean(np.array(fi_list)), 3)
    accuracy = round(sum(TP_list) / sum(pred_sum) * 100, 3)
    
    return accuracy, recall, precision, f1



# 混同行列を作る、行が正解、列が予測
def make_mixed_matrix(df, target_name, RATE_list):
    ans_list = df.loc[:,target_name]
    pred_list = df.loc[:,'予測']
    df_matrix = pd.crosstab(ans_list, pred_list)
    df_matrix_sort = df_matrix.reindex(index=RATE_list,
                                       columns=RATE_list)
    
    t_list = []
    p_list = []
    for i, c in zip(df_matrix_sort.index, df_matrix_sort.columns):
        t_list.append('正解: ' +  str(i))
        p_list.append('予測: ' +  str(c))
    
    df_matrix_sort.set_axis(t_list, axis="index", inplace=True)
    df_matrix_sort.set_axis(p_list, axis="columns", inplace=True)
    
    # num = len(RATE_list)
    # matrix = np.zeros((num,num))
    # ans_list = df.loc[:,target_name]
    # pred_list = df.loc[:,'予測']
    # for ans, pred in zip(ans_list, pred_list):
    #     matrix[ans, pred] += 1
    # df_matrix = pd.DataFrame(matrix)
    
    # t_list = []
    # p_list = []
    # for r in RATE_list:
    #     t_list.append('正解: ' +  str(r))
    #     p_list.append('予測: ' +  str(r))
    
    # df_matrix.set_axis(t_list, axis="index", inplace=True)
    # df_matrix.set_axis(p_list, axis="columns", inplace=True)
    
    return df_matrix_sort





# 読み込みRAMリストから優先順位をつけて選択する
def read_data(data, value, RAM_names, data_set):
    RAM = RAM_names.at[value, data_set]
    if ',' in RAM:
        RAM_list = RAM.split(',')
        data_read = 'None'
        for i in range(len(RAM_list)):
            try:
                data_read = data[RAM_list[i]]
                break
            except:
                pass
    else:
        data_read = data[RAM]
        
    return np.array(data_read)



def band_FFT(df, time, min_freq, max_freq):
    signal = df.values
    t = time.values

    # サンプリング周波数とサンプル数を設定
    fs = 1 / (t[1] - t[0])
    N = signal.shape[0]

    # フーリエ変換を実行
    spectrum = np.fft.fft(signal)

    # 振幅スペクトルをコピーして選択した帯域以外をゼロにする
    filtered_spectrum = spectrum.copy()
    filtered_spectrum[(np.abs(fs * np.fft.fftfreq(N)) < min_freq) | (np.abs(fs * np.fft.fftfreq(N)) > max_freq)] = 0

    # 逆フーリエ変換を実行して特定の周波数帯域の信号を取り出す
    filtered_signal = np.real(np.fft.ifft(filtered_spectrum))
    
    return filtered_signal



def cal_RATE_070D(data, resolution, st, en):
    # rate = sum(data[st:en]) / len(data[st:en])
    rate = max(data[st:en])
    # マーカーの測定値を官能評点に変換
    if 0.4 - resolution < rate < 0.4 + resolution:
        I_RATE = '2.5'
        # 0.275 ~ 0.525
    elif 0.75 - resolution < rate < 0.75 + resolution:
        I_RATE = '3-'
        # 0.625 ~ 0.875
    elif 1.1 - resolution < rate < 1.1 + resolution:
        I_RATE = '3'
        # 0.975 ~ 1.225
    elif 1.4 - resolution < rate < 1.4 + resolution:
        I_RATE = '3+'
        # 1.275 ~ 1.525
    elif 1.65 - resolution < rate < 1.7 + resolution:
        I_RATE = '3.5'
        # 1.525 ~ 1.825
    else:
        I_RATE = 'unknown'
        
    return [I_RATE, rate]





def split_files_train_test(data_set, learn_type):
    # ファイル名と官能評点を読み込み
    df_file_RATE = pd.read_csv('result/result_G2RATE/ファイル名と官能評点_' + data_set + '.csv', header=None)
    
    # 目的変数の調整（偏りのある場合はリストから読み込んだ目的変数のファイルを除外する）
    if learn_type == 'G2Rate':
        target_remove_list = pd.read_csv('setting/target_remove/target_remove_' + data_set + '.csv', header=None).values.reshape(-1).tolist()
        tmp_remove = ['迷い'] + target_remove_list
        drop_list = []
        for d in range(len(df_file_RATE)):
            for r in tmp_remove:
                if r in df_file_RATE.iat[d, 1]:
                    drop_list.append(df_file_RATE.index.to_numpy()[d])
        df_file_RATE.drop(drop_list, inplace=True)
        df_file_RATE.sort_values(by=df_file_RATE.columns[0], inplace=True)
        df_file_RATE.reset_index(drop=True, inplace=True)
    elif learn_type == 'To2G':
        pass
    sample_num = len(df_file_RATE)
    print('サンプルの個数 : {}個'.format(sample_num))
    RATE = list(df_file_RATE.values[:,1])
    RATE_list = sort_RATE_list(list(set(RATE)), data_set)
    for r in RATE_list:
        print('官能評点 {} の個数 : {} 個　（{:.2f}%）'.format(r, RATE.count(r), RATE.count(r)/len(RATE)*100))
    print()
    
    split_list = []
    for i in range(len(df_file_RATE)):
        filename = df_file_RATE.iat[i,0].split('_')[0]
        num = df_file_RATE.iat[i,0].split('_')[1]
        sft = df_file_RATE.iat[i,0].split('_')[2]
        rate = df_file_RATE.iat[i,1]
        split_list.append([filename, num, sft, rate])
    df_split = pd.DataFrame(split_list, columns=['ファイル名', '何番目', '変速種類', '官能評点'])
    df_split.sort_values(by='ファイル名', inplace=True)
    # print(df_split)
    file_num = list(set(df_split['ファイル名'].values))
    test_filename = []
    test_rate_list = []
    for f in file_num:
        # print(f)
        df_tmp = df_split[df_split['ファイル名']==f]
        rate_unique = list(set(df_tmp['官能評点'].values))
        sft_unique = list(set(df_tmp['変速種類'].values))
        rate_count_list = []
        for u in rate_unique:
            for s in sft_unique:
                df_bool = (df_tmp['官能評点'] == u) & (df_tmp['変速種類'] == s)
                rate_count_list.append([u, s, df_bool.sum()])
                df_count = pd.DataFrame(rate_count_list, columns=['官能評点', '変速種類', '個数'])
        # print(df_count)
        df_num = df_count[df_count['個数']>=3]
        df_num.reset_index(drop=True, inplace=True)
        # print(df_num)
        if len(df_num) > 0:
            for n in range(len(df_num)):
                test_rate = df_num.at[n,'官能評点']
                test_sft = df_num.at[n,'変速種類']
                df_test = df_tmp[(df_tmp['官能評点']==test_rate)&(df_tmp['変速種類']==test_sft)]
                df_test.sort_values(by='何番目', ascending=False, inplace=True)
                df_test.reset_index(drop=True, inplace=True)
                add_filename = f + '_' + df_test.at[0, '何番目'] + '_' + test_sft
                test_filename.append(add_filename)
                test_rate_list.append(test_rate)
                if len(df_test) >= 10000:
                    add_filename = f + '_' + df_test.at[1, '何番目'] + '_' + test_sft
                    test_filename.append(add_filename)
                    test_rate_list.append(test_rate)
    
    df_split_test_files = pd.concat([pd.DataFrame(test_filename), pd.DataFrame(test_rate_list)], axis=1)
    df_split_test_files.set_axis(['ファイル名', '官能評点'], axis=1, inplace=True)
    test_num = len(df_split_test_files)
    print('テスト用に分離する個数 : {}個'.format(test_num))
    
    RATE = list(df_split_test_files.values[:,1])
    RATE_list = sort_RATE_list(list(set(RATE)), data_set)
    for r in RATE_list:
        print('官能評点 {} の個数 : {} 個　（{:.2f}%）'.format(r, RATE.count(r), RATE.count(r)/len(RATE)*100))
    print()
    
    if test_num > int(sample_num*0.1):
        print('分離する個数が多いのでテスト用ファイルをランダムにドロップ')
        # 乱数を初期化
        random.seed(0)
        drop_rate = int(sample_num*0.1) / test_num
        index_randam_drop = []
        for r in RATE_list:
            df_r = df_split_test_files[df_split_test_files[df_split_test_files.columns[1]] == r]
            index_r = list(df_r.index)
            # print(index_r)
            random.shuffle(index_r)
            index_random = index_r[int(len(index_r) * drop_rate):]
            # print(index_random)
            index_randam_drop += index_random
        
        # print(index_randam_drop)
        df_split_test_files.drop(index_randam_drop, inplace=True)
        df_split_test_files.sort_values(by='ファイル名', inplace=True)
        df_split_test_files.reset_index(drop=True, inplace=True)
        
        test_num = len(df_split_test_files)
        print(' -テスト用に分離する個数 : {}個'.format(test_num))
        RATE = list(df_split_test_files.values[:,1])
        RATE_list = sort_RATE_list(list(set(RATE)), data_set)
        for r in RATE_list:
            print(' -官能評点 {} の個数 : {} 個　（{:.2f}%）'.format(r, RATE.count(r), RATE.count(r)/len(RATE)*100))

    if learn_type == 'G2Rate':
        df_split_test_files.to_csv('train_data/G2RATEテスト用に学習対象外とする切り出しデータのリスト_' + data_set + '.csv', index=False, encoding='utf_8_sig')
    elif learn_type == 'To2G':
        df_split_test_files.drop('官能評点', axis=1, inplace=True)
        df_split_test_files.to_csv('train_data/To2Gテスト用に学習対象外とする切り出しデータのリスト_' + data_set + '.csv', index=False, header=False, encoding='utf_8_sig')


