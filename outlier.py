import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns



####### MT法で外れ値検出 #######
def outlier_MT(df, columns_list, e_N, t_N):
    # 使うデータを抜き出す
    x = df.loc[:,[columns_list[e_N],columns_list[t_N]]]
    x1 = df.loc[:,columns_list[e_N]]
    x2 = df.loc[:,columns_list[t_N]]
    
    # 多変量連関図で生データを吟味
    sns.pairplot(x)
    
    # 各データの平均値を求め、平均値を引き算する
    divide = 2       # 状態量の数=使う変数
    N, _ = x.shape   # データ数
    p = 0.95         # マハラノビス距離p=0.95で2σ
    div = 50         # Mt楕円の分割数
    R = np.zeros((divide,divide))    # 相関行列の型の設定
    invR = np.zeros((divide,divide)) # 相関行列の逆行列の型の設定
    
    # 各状態量から平均値を引く x の大きさは、(N,divide)
    x = np.array(x,dtype="float64")    # xをnumpyのarray形式に変換
    xx = np.copy(x)                    # xをコピーして、xxにする
    x_mean = np.zeros(divide)          # x_mean(平均値)の型の設定
    for i in range(divide):
        x_mean[i] = np.mean(x[:,i])
        print('X[',i,'] 平均 =',x_mean[i])
        for j in range(N):
            xx[j,i] = x[j,i] - x_mean[i]
    
    # 各状態量を標準偏差で割る
    x_std = np.zeros(divide)
    for i in range(divide):
        x_std[i] = np.std(x[:,i])
        print('X[',i,'] 標準偏差 =',x_std[i])
        for j in range(N):
            xx[j,i] = xx[j,i] / x_std[i]
    
    # 相関係数行列とその逆行列を求める
    R = np.corrcoef(xx.transpose())
    invR = np.linalg.inv(R)
    print('R:')
    print(R)
    print('\ninvR:')
    print(invR)
    
    # マハラノビス距離（データの行列＊相関係数行列の逆行列＊データの転置行列 のルート）を求める
    md = []
    for j in range(N):
        d0 = xx[j,:].reshape((1,divide))
        d1 = np.dot(d0,invR)
        d2 = np.dot(d1,d0.T)
        d2 = math.sqrt(d2)
        md.append(d2)
    
    # 外れ値特定
    df_md = pd.concat([df.loc[:,[columns_list[e_N],columns_list[t_N]]],
                        pd.DataFrame(md, columns=['Mahalanobis distance'])],
                        axis=1)
    # for j in range(N):
    #     if md[j] >= 2.448:
    #         print((j+1),df.iloc[j,e_N],df.iloc[j,t_N],md[j])
    
    # 2σ（マハラノビス距離＝2.448）の楕円を求める
    curve_c = np.zeros((2,div+1))
    low = np.corrcoef(xx[:,0],xx[:,1])[0,1]
    
    for i in range(div+1):
        r = (-2*(1-low**2)*np.log(1-p)/(1-2*low*np.sin(i*2*np.pi/div)*np.cos(i*2*np.pi/div)))**0.5
        curve_c[0,i] = x_mean[0] + x_std[0]*r*np.cos(i*2*np.pi/div)
        curve_c[1,i] = x_mean[1] + x_std[1]*r*np.sin(i*2*np.pi/div)
    
    # 可視化
    plt.subplot(1,1,1)
    plt.scatter(x1, x2, c="green", s=50,label="")
    plt.xlabel(df.columns[e_N])
    plt.ylabel(df.columns[t_N])
    plt.plot(curve_c[0],curve_c[1],c="c",label="MD=2.448")
    plt.legend(loc='upper left')
    plt.show()
    
    return df_md



####### 1クラスSVMで外れ値検出 #######
# グリッドサーチ実行
def outlier_OCSVM1(df, OCSVM_list):
    X = df.loc[:,OCSVM_list]
    
    # 多変量連関図で生データを吟味
    sns.pairplot(X)
    
    # 説明変数を標準化する
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    # ハイパーパラメータgammaの最適化 
    # log10単位で大きく変えて最適範囲を探す
    np_gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    df2 = pd.DataFrame()
    for np_gamma in np_gammas:
        mdl = svm.OneClassSVM(nu=0.05, kernel="rbf",gamma=np_gamma)
        mdl.fit(X_std)
        pred = mdl.predict(X_std)
        pred_err = [i<0 for i in pred]
        temp = pd.DataFrame({'gamma':np_gamma ,'偽陽性率':sum(pred_err)/len(pred_err)},index=[np_gamma,])
        print("gamma:{0}".format(np_gamma), "異常件数:{0}".format(sum(pred_err)),
                "偽陽性率:{0}".format(sum(pred_err)/len(pred_err)))
        df2 = pd.concat([df2,temp])
    
    gamma_best_index = df2['偽陽性率'].idxmin()
    gamma_best = df2.loc[gamma_best_index, 'gamma']

    #　gammaと偽陽性率の推移を可視化
    x = df2['gamma']
    y = df2['偽陽性率']
    plt.plot(x, y)
    plt.xscale('log')
    plt.xlabel('gamma')
    plt.ylabel('rate')
    plt.grid()
    plt.hlines([0.05], df2['gamma'].min() ,df2['gamma'].max() , "red")
    plt.show()

    # ハイパーパラメータgammaの変化による偽陽性率の推移を確認
    # 上で見つけた範囲で小刻みにgammaを変え、ズレの小さいgammaを探す
    # gamma 0.01から、0.01刻みで 2まで増やして偽陽性率の推移を確認する
    df3 = pd.DataFrame()
    ini = gamma_best
    times = 200
    for i in range(times):
        mdl = svm.OneClassSVM(nu=0.0001, kernel="rbf",gamma=(i+1)*ini)
        mdl.fit(X_std)
        pred = mdl.predict(X_std)
        pred_err = [i<0 for i in pred]
        temp = pd.DataFrame({'gamma':(i+1)*ini ,'偽陽性率':sum(pred_err)/len(pred_err)},index=[i,])
        df3 = pd.concat([df3,temp])
    
    # gamma による偽陽性率の推移を可視化する
    x = df3['gamma']
    y = df3['偽陽性率']
    plt.plot(x, y)
    plt.xlabel('gamma')
    plt.ylabel('rate')
    plt.grid()
    plt.hlines([0.05], df3['gamma'].min() ,df3['gamma'].max() , "red")
    plt.show()

    return gamma_best
    


# パラメータ最適値で1クラスSVM実行
def outlier_OCSVM2(df, OCSVM_list, gamma_best):
    X = df.loc[:,OCSVM_list]
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    # 作成モデルでデータを予測する
    mdl = svm.OneClassSVM(nu=0.05, kernel="rbf",gamma=gamma_best)
    mdl.fit(X_std)
    pred = mdl.predict(X_std)
    
    pred_err = [i<0 for i in pred]
    print("異常件数:{0}".format(sum(pred_err)))
    print("偽陽性率:{0}".format(sum(pred_err)/len(pred_err)))
    
    # 予測したデータの確認
    df4 = pd.concat([X,pd.DataFrame(X_std,columns=[OCSVM_list[0]+'_std',OCSVM_list[1]+'_std'])],axis=1) 
    print(df4[pred_err])
    
    # OneClassSVM グラフ描画
    df5 = pd.DataFrame(X_std,columns=[OCSVM_list])
    plt.scatter(df5.iloc[:,0], df5.iloc[:,1])
    # mesh
    x_min, x_max = df5.iloc[:,0].min() - 1, df5.iloc[:,0].max() + 1
    y_min, y_max = df5.iloc[:,1].min() - 1, df5.iloc[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    # contour
    Z = mdl.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.show()
    
    # gamma値の違いによるグラフの変化
    gammna_list = np.array([gamma_best/1000, gamma_best/100, gamma_best/10,
                            gamma_best/2, gamma_best, gamma_best*5,
                            gamma_best*10, gamma_best*100, gamma_best*1000])
    
    fig = plt.figure(figsize = (15, 9))
    ax = [ ]
    for i in np.arange(1,len(gammna_list)+1):
        ax_add = fig.add_subplot(3, 3, i)
        ax.append(ax_add)
    # mesh
    x_min, x_max = df5.iloc[:,0].min() - 1, df5.iloc[:,0].max() + 1
    y_min, y_max = df5.iloc[:,1].min() - 1, df5.iloc[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    for i in np.arange(0, len(gammna_list)):
        ax[i].scatter(df5.iloc[:,0], df5.iloc[:,1])
        # 作成モデルでデータを予測する
        mdl = svm.OneClassSVM(nu=0.05, kernel="rbf",gamma=gammna_list[i])
        mdl.fit(X_std)
        pred = mdl.predict(X_std)
        pred_err = [i<0 for i in pred]
        # contour
        Z = mdl.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax[i].contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
        title_name = 'gamma = ' + gammna_list[i].astype(str)
        ax[i].set_title(title_name)
        ax[i].set_xlabel(df5.columns[0])
        ax[i].set_ylabel(df5.columns[1])
    plt.show()














