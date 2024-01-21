import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from Scripts.Utility import timedelta_to_hms



# 切り出し
def plot_slice_data(tmp_data, slice_st, slice_en, filename, RAM_names, project, data_set):
    file_path = 'process_data/slice_data_plot_' + data_set + '/' + filename + '.png'
    if data_set == '070D_MONUP' or data_set == '070D_MOFFDWN':
        df_RAMconv = pd.read_csv('process_data/学習用RAM名変換リスト_' + data_set + '.csv', header=0, index_col=0)
        time = tmp_data[df_RAMconv.at[RAM_names.at['ECUTIME', project],'秘匿化RAM名']].values
        pap = tmp_data[df_RAMconv.at[RAM_names.at['pap', project],'秘匿化RAM名']].values
        G = tmp_data[df_RAMconv.at[RAM_names.at['G', project],'秘匿化RAM名']].values
        nextgear = tmp_data[df_RAMconv.at[RAM_names.at['nextgear', project],'秘匿化RAM名']].values
        sftjdg = tmp_data[df_RAMconv.at[RAM_names.at['sftjdg', project],'秘匿化RAM名']].values
        xsft = tmp_data[df_RAMconv.at[RAM_names.at['xsft', project],'秘匿化RAM名']].values
        xina = tmp_data[df_RAMconv.at[RAM_names.at['xina', project],'秘匿化RAM名']].values
        Ne = tmp_data[df_RAMconv.at[RAM_names.at['Ne', project],'秘匿化RAM名']].values
        Nt = tmp_data[df_RAMconv.at[RAM_names.at['Nt', project],'秘匿化RAM名']].values
        Nm = tmp_data[df_RAMconv.at[RAM_names.at['Nm', project],'秘匿化RAM名']].values
        NOGEAR_before = tmp_data[df_RAMconv.at['NOGEAR_before', '秘匿化RAM名']].values
        NOGEAR_after = tmp_data[df_RAMconv.at['NOGEAR_after', '秘匿化RAM名']].values
        G_Butter10 = tmp_data[df_RAMconv.at['G_Butter10','秘匿化RAM名']].values
        G_Butter40 = tmp_data[df_RAMconv.at['G_Butter40','秘匿化RAM名']].values
        
        fig = plt.figure(figsize=(21,16))
        plt.subplot(5,1,1)
        plt.title(label='【学習領域切り出し】 {}'.format(filename), fontsize=16)
        plt.plot(time, pap, linewidth=2, color='black', label='pap')
        plt.axvspan(time[0], time[slice_st], color='gray', alpha=0.2)
        plt.axvspan(time[slice_en], time[-1], color='gray', alpha=0.2)
        plt.ylim(np.min(pap)-10, np.max(pap)+10)
        plt.legend(loc='upper right', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.subplot(5,1,2)
        plt.plot(time, nextgear, linewidth=3, color='green', label='nextgear')
        plt.plot(time, xina, linewidth=2, color='cyan', label='xina')
        plt.plot(time, sftjdg, linewidth=2, color='red', label='sftjdg')
        plt.plot(time, xsft, linewidth=1, color='black', label='xsft')
        plt.axvspan(time[0], time[slice_st], color='gray', alpha=0.2)
        plt.axvspan(time[slice_en], time[-1], color='gray', alpha=0.2)
        plt.legend(loc='upper right', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.subplot(5,1,3)
        plt.plot(time, G, linewidth=1, color='black', label='G')
        plt.plot(time, G_Butter10, linewidth=2, color='red', label='Butterworth 10Hz G')
        plt.axvspan(time[0], time[slice_st], color='gray', alpha=0.2)
        plt.axvspan(time[slice_en], time[-1], color='gray', alpha=0.2)
        plt.legend(loc= 'upper right', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.subplot(5,1,4)
        plt.plot(time, G, linewidth=1, color='black', label='G')
        plt.plot(time, G_Butter40, linewidth=2, color='orange', label='Butterworth 40Hz G')
        plt.axvspan(time[0], time[slice_st], color='gray', alpha=0.2)
        plt.axvspan(time[slice_en], time[-1], color='gray', alpha=0.2)
        plt.legend(loc= 'upper right', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.xlabel('time [s]')
        plt.grid(True)
        plt.subplot(5,1,5)
        plt.plot(time, Ne, linewidth=2, color='red', label='Ne')
        plt.plot(time, Nt, linewidth=2, color='green', label='Nt')
        plt.plot(time, Nm, linewidth=2, color='orange', label='Nm')
        plt.plot(time, NOGEAR_before, linewidth=1, color='black', linestyle='dotted', label='NOGEAR before')
        plt.plot(time, NOGEAR_after, linewidth=1, color='black', linestyle='dotted', label='NOGEAR after')
        plt.axvspan(time[0], time[slice_st], color='gray', alpha=0.2)
        plt.axvspan(time[slice_en], time[-1], color='gray', alpha=0.2)
        plt.legend(loc='upper right', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.close()
        fig.savefig(file_path)
        


def plot_slice_data_test(tmp_data, slice_st, slice_en, filename, RAM_names, data_set):
    file_path = 'result/result_G2RATE/【テスト】slice_data_plot_' + data_set + '/' + filename + '.png'
    if data_set == '070D_MONUP' or data_set == '070D_MOFFDWN':
        ####### 070D未対応 （学習用のデータセットで切り出し済のものを使う）
        df_RAMconv = pd.read_csv('train_data/G2Rate学習用RAM名変換リスト_' + data_set + '.csv', header=0, index_col=0)
        time = tmp_data[df_RAMconv.at[RAM_names.at['ECUTIME', data_set],'秘匿化RAM名']].values
        pap = tmp_data[df_RAMconv.at[RAM_names.at['pap', data_set],'秘匿化RAM名']].values
        G = tmp_data[df_RAMconv.at[RAM_names.at['G', data_set],'秘匿化RAM名']].values
        nextgear = tmp_data[df_RAMconv.at[RAM_names.at['nextgear', data_set],'秘匿化RAM名']].values
        sftjdg = tmp_data[df_RAMconv.at[RAM_names.at['sftjdg', data_set],'秘匿化RAM名']].values
        xsft = tmp_data[df_RAMconv.at[RAM_names.at['xsft', data_set],'秘匿化RAM名']].values
        xina = tmp_data[df_RAMconv.at[RAM_names.at['xina', data_set],'秘匿化RAM名']].values
        Ne = tmp_data[df_RAMconv.at[RAM_names.at['Ne', data_set],'秘匿化RAM名']].values
        Nt = tmp_data[df_RAMconv.at[RAM_names.at['Nt', data_set],'秘匿化RAM名']].values
        Nm = tmp_data[df_RAMconv.at[RAM_names.at['Nm', data_set],'秘匿化RAM名']].values
        NOGEAR_before = tmp_data[df_RAMconv.at['NOGEAR_before', '秘匿化RAM名']].values
        NOGEAR_after = tmp_data[df_RAMconv.at['NOGEAR_after', '秘匿化RAM名']].values
        G_Butter10 = tmp_data[df_RAMconv.at['G_Butter10','秘匿化RAM名']].values
        G_Butter40 = tmp_data[df_RAMconv.at['G_Butter40','秘匿化RAM名']].values
        
        fig = plt.figure(figsize=(21,16))
        plt.subplot(5,1,1)
        plt.title(label='【学習領域切り出し】 {}'.format(filename), fontsize=16)
        plt.plot(time, pap, linewidth=2, color='black', label='pap')
        plt.axvspan(time[0], time[slice_st], color='gray', alpha=0.2)
        plt.axvspan(time[slice_en], time[-1], color='gray', alpha=0.2)
        plt.ylim(np.min(pap)-10, np.max(pap)+10)
        plt.legend(loc='upper right', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.subplot(5,1,2)
        plt.plot(time, nextgear, linewidth=3, color='green', label='nextgear')
        plt.plot(time, xina, linewidth=2, color='cyan', label='xina')
        plt.plot(time, sftjdg, linewidth=2, color='red', label='sftjdg')
        plt.plot(time, xsft, linewidth=1, color='black', label='xsft')
        plt.axvspan(time[0], time[slice_st], color='gray', alpha=0.2)
        plt.axvspan(time[slice_en], time[-1], color='gray', alpha=0.2)
        plt.legend(loc='upper right', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.subplot(5,1,3)
        plt.plot(time, G, linewidth=1, color='black', label='G')
        plt.plot(time, G_Butter10, linewidth=2, color='red', label='Butterworth 10Hz G')
        plt.axvspan(time[0], time[slice_st], color='gray', alpha=0.2)
        plt.axvspan(time[slice_en], time[-1], color='gray', alpha=0.2)
        plt.legend(loc= 'upper right', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.subplot(5,1,4)
        plt.plot(time, G, linewidth=1, color='black', label='G')
        plt.plot(time, G_Butter40, linewidth=2, color='orange', label='Butterworth 40Hz G')
        plt.axvspan(time[0], time[slice_st], color='gray', alpha=0.2)
        plt.axvspan(time[slice_en], time[-1], color='gray', alpha=0.2)
        plt.legend(loc= 'upper right', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.xlabel('time [s]')
        plt.grid(True)
        plt.subplot(5,1,5)
        plt.plot(time, Ne, linewidth=2, color='red', label='Ne')
        plt.plot(time, Nt, linewidth=2, color='green', label='Nt')
        plt.plot(time, Nm, linewidth=2, color='orange', label='Nm')
        plt.plot(time, NOGEAR_before, linewidth=1, color='black', linestyle='dotted', label='NOGEAR before')
        plt.plot(time, NOGEAR_after, linewidth=1, color='black', linestyle='dotted', label='NOGEAR after')
        plt.axvspan(time[0], time[slice_st], color='gray', alpha=0.2)
        plt.axvspan(time[slice_en], time[-1], color='gray', alpha=0.2)
        plt.legend(loc='upper right', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.close()
        fig.savefig(file_path)





# 特徴量をプロット
def plot_box(explanatory_list, target_name, I_df2, class_mapping_inverse, DirName):
    start_time = datetime.datetime.now()

    if len(explanatory_list) > 0:
        df_Feat = I_df2[explanatory_list + [target_name]].copy()
        df_Feat.sort_values(target_name, ascending=False, inplace=True)
        df_Feat[target_name] = df_Feat[target_name].map(class_mapping_inverse)
        for k, feat in enumerate(explanatory_list):
            if k == -1:
                print(' -分布表示する特徴量数が{}を越えたので作成を強制停止します'.format(k))
                break
            fig = plt.figure(figsize=(16,9))
            plt.title(feat)
            sns.boxplot(x=target_name, y=feat, data=df_Feat, showfliers=False)
            sns.stripplot(x=target_name, y=feat, data=df_Feat, jitter=True, color='black')
            plt.close()
            fig.savefig('process_data/' + DirName + '/' + feat + '.png')
    calc_time = datetime.datetime.now() - start_time
    h, m, s = timedelta_to_hms(calc_time)
    print(' -{}のプロット出力時間 : {:02d}hr {:02d}min {:.1f}sec'.format(DirName, h, m, s))





# 寄与度の高い特徴量をプロット
def plot_significant_feature(feature_importance_LGBM, target_name, I_df, class_mapping_inverse, savedir):
    feature_importance_LGBM.reset_index(drop=True, inplace=True)
    feature_name_list = [target_name] + list(feature_importance_LGBM['feature_name'])
    # 特徴量の分布
    df_HighContFeat = I_df.loc[:, feature_name_list]
    df_HighContFeat.sort_values(target_name, ascending=False, inplace=True)
    df_HighContFeat[target_name] = df_HighContFeat[target_name].map(class_mapping_inverse)
    for k, feat in enumerate(list(feature_importance_LGBM['feature_name'])):
        if k >= 100:
            print('分布表示する特徴量数が{}を越えたので作成を強制停止します'.format(k))
            break
        fig = plt.figure(figsize=(16,9))
        plt.title(feat)
        sns.boxplot(x=target_name, y=feat, data=df_HighContFeat, showfliers=False)
        sns.stripplot(x=target_name, y=feat, data=df_HighContFeat, jitter=True, color='black')
        plt.close()
        fig.savefig(savedir + '/寄与度の高い特徴量分布/寄与度' + str(k+1) + '_' + feat + '.png')





# 寄与度上位の特徴量を2軸プロット
def plot_2axis(feature_importance_LGBM, target_name, df_LGBM, RATE_list, savedir):
    feature_importance_LGBM.reset_index(drop=True, inplace=True)
    feature_name_list = list(feature_importance_LGBM['feature_name'])
    df_HighContFeat = df_LGBM[[target_name, '予測'] + feature_name_list]
    
    feat_num = min(5, len(feature_name_list))
    feats = []
    for i in range(feat_num):
        feats.append(feature_name_list[i])
    
    df_list = []
    for r in RATE_list:
        df_tmp = df_HighContFeat[df_HighContFeat[target_name]==r]
        df_list.append(df_tmp)
    
    def plot_a_b_2axsis(df_list, a, b, feat_a, feat_b):
        fig = plt.figure(figsize=(16,9))
        for i in range(len(df_list)):
            if RATE_list[i] == '3.5':
                color = 'black'
            elif RATE_list[i] == '3+':
                color = 'purple'
            elif RATE_list[i] == '3':
                color = 'blue'
            elif RATE_list[i] == '3-':
                color = 'cyan'
            elif RATE_list[i] == '2.5':
                color = 'red'
            else:
                color = 'black'
            plt.scatter(df_list[i][feat_a], df_list[i][feat_b], s=80, c=color, label='官能{}'.format(RATE_list[i]))
            for k in range(len(df_list[i])):
                df_list[i].reset_index(drop=True, inplace=True)
        plt.tick_params(labelsize=14)
        plt.xlabel(feat_a, fontsize=16)
        plt.ylabel(feat_b, fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.close()
        fig.savefig(savedir + '/寄与度上位の特徴量2軸プロット/{}位と{}位.png'.format(a, b))
    
    for n in range(feat_num-1):
        for ny in range(n+1, feat_num):
                plot_a_b_2axsis(df_list, n+1, ny+1, feats[n], feats[ny])
    





