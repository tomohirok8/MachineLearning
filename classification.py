import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import itertools
import matplotlib.pyplot as plt
import seaborn as sns



####### 階層的クラスター分析で層別 #######
def hierarchical_cluster_analysis(df, explanatory_list, max_cluster):
    # Xに説明変数をセットする。
    X = df.loc[:,explanatory_list]
    
    # 説明変数を標準化する
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    # クラスタリング実施（距離の近いサンプル同士を結合してゆく）
    linkage_array = linkage(X_std, method='ward', metric="euclidean")
    
    # クラスタ最大数を設定してどのクラスタに属するを決定する
    cluster = fcluster(linkage_array, max_cluster , criterion='maxclust')
    
    # 実測値との比較のため元データに予測した結果を結合
    df2 = pd.concat([df,pd.DataFrame(cluster, columns=['Cluster No'])],axis=1) 
    
    # デンドログラムの表示
    plt.figure(figsize=(40, 20))
    plt.rcParams['font.size'] = 20
    plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
    plt.xlabel('Observation Points', fontsize=20)
    plt.ylabel('Distance', fontsize=20)
    dendrogram(linkage_array, color_threshold=40, above_threshold_color='black')
    plt.show()
    
    # クラスター毎に標準化された平均
    df3 = pd.concat([pd.DataFrame(X_std,columns=explanatory_list),
                     pd.DataFrame(cluster,columns=['Cluster No'])],axis=1)
    
    # クラスター毎の平均値比較グラフを描画する
    df4 = df3.groupby('Cluster No').mean()
    plt.figure(figsize=(15, 7))
    plt.rcParams['font.size'] = 12
    x = list(range(len(df4.columns)))
    for i in range(len(df4)):
        y = df4.T.iloc[:,i]
        plt.plot(x,  y, marker="o")
    labels = df4.columns
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel('average', fontsize=12)
    plt.legend(''.join([str(n) for n in list(np.array(list(range(1,max_cluster+1))))]))
    plt.grid(True)
    plt.show()
    
    return df2



####### k-means法で層別 #######
def kmeans_classification(df, explanatory_list, n_cluster):
    # Xに説明変数をセットする。
    X = df.loc[:,explanatory_list]
    
    # 説明変数を標準化する
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    # クラスタ内誤差平方和を計算（小さいほど「歪みのない良いモデル」）
    distortions = []
    for i in range(1, 50):
        mdl = KMeans(n_clusters=i)
        mdl.fit(X_std)
        distortions.append(mdl.inertia_)
        # print('Distortion: %.2f' % mdl.inertia_)
        
    plt.plot(range(1, 50), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.grid(True)
    plt.show()
    
    # 結果のサマリ
    np.random.seed(0) #Kmeans法は初期クラスタを乱数で決めるので乱数の種を固定 ⇒ 固定しないと結果が毎回異なる
    mdl = KMeans(n_clusters=n_cluster)
    mdl.fit(X_std)
    pred = mdl.predict(X_std)
    
    # クラスター毎に標準化された平均
    df2 = pd.concat([pd.DataFrame(X_std,columns=explanatory_list),
                     pd.DataFrame(pred,columns=['Cluster No'])],axis=1)
    
    # クラスター毎の平均値比較グラフを描画する
    df3 = df2.groupby('Cluster No').mean()
    plt.figure(figsize=(15, 7))
    x = list(range(len(df3.columns)))
    for i in range(len(df3)):
        y = df3.T.iloc[:,i]
        plt.plot(x,  y, marker="o")
    labels = df3.columns
    plt.xticks(x, labels)
    plt.ylabel('average')
    plt.legend(''.join([str(n) for n in list(np.array(list(range(1,n_cluster+1))))]))
    plt.grid(True)
    plt.show()
    
    # クラスタリングしたデータの確認
    df4 = pd.concat([df, pd.DataFrame(pred,columns=['Cluster No'])],axis=1) # 実測値との比較のため元データに予測した結果を結合
    
    return df4



####### 混合ガウス分布で層別 #######
def GaussianMixtureModel_classification(df, explanatory_list):
    # Xに説明変数をセットする。
    X = df.loc[:,explanatory_list]

    # 説明変数を標準化する
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    # BIC(ベイズ情報量基準)で最適モデルを探索する
    # 情報量基準とは　モデルの良さを測るもの
    # BICが低いほど、説明変数が少ないか、適合度が高い、あるいはその両方を示している＝良いモデルと思って良い
    # BICが最小となるガウス分布の最良となるモデルを探索
    lowestBIC = np.infty
    bic = []
    nComponentsRange = range(1, 6) # コンポーネント数の範囲 1～5
    cvTypes = ['spherical', 'tied', 'diag', 'full']
    # Covariance typeはクラスターの形を設定
    # 'spherical' 各構成要素には独自の分散があります,
    # 'tied' すべての構成要素が同じ一般的な共分散行列を共有する
    # 'diag' 各構成要素には独自の対角共分散行列があります
    # 'full' 各構成要素には独自の一般共分散行列があります　制限なし
    for cvType in cvTypes:
        for nComponents in nComponentsRange:
            # Fit a Gaussian mixture with EM
            mdl = mixture.GaussianMixture(n_components=nComponents,covariance_type=cvType, random_state=777)
            mdl.fit(X_std)
            bic.append(mdl.bic(X_std))
            
            if bic[-1] < lowestBIC: # 最小BICの更新
                lowestBIC = bic[-1]
                bestMdl = mdl
    
    # 最小BICの更新
    print("Best covariance_type =",bestMdl.covariance_type) 
    print("Best n_components =",bestMdl.n_components)
    
    # BICスコアをプロット
    bic = np.array(bic)
    colorIter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange'])
    plt.figure(figsize=(8, 6),dpi=100)
    bars = []
    ax = plt.subplot(111)
    for i, (cvType, color) in enumerate(zip(cvTypes, colorIter)):
        xpos = np.array(nComponentsRange) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(nComponentsRange):
                                      (i + 1) * len(nComponentsRange)],
                            width=.2, color=color))
    plt.xticks(nComponentsRange)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(nComponentsRange)) + .65 +\
        .2 * np.floor(bic.argmin() / len(nComponentsRange))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    ax.set_xlabel('Number of components')
    ax.legend([b[0] for b in bars], cvTypes)
    
    # クラスタリングを実施
    np.random.seed(0) # Kmeans法は初期クラスタを乱数で決めるので乱数の種を固定 ⇒ 固定しないと結果が毎回異なる
    # もっとも情報量基準が良かった条件でクラスタリングを実行
    mdl = mixture.GaussianMixture(n_components=bestMdl.n_components, covariance_type=bestMdl.covariance_type)
    mdl.fit(X_std)
    pred = mdl.predict(X_std)
    
    df2 = pd.concat([pd.DataFrame(X_std,columns=explanatory_list),
                     pd.DataFrame(pred,columns=['Cluster No'])],axis=1)
    
    # クラスター毎の平均値比較グラフを描画する
    df3 = df2.groupby('Cluster No').mean()
    plt.figure(figsize=(15, 7))
    x = list(range(len(df3.columns)))
    for i in range(len(df3)):
        y = df3.T.iloc[:,i]
        plt.plot(x,  y, marker="o")
    labels = df3.columns
    plt.xticks(x, labels)
    plt.ylabel('average')
    plt.legend('12345')
    plt.grid(True)
    plt.show()
    
    # 元データにクラスタ番号を付与
    df4 = pd.concat([df, pd.DataFrame(pred,columns=['Cluster No'])],axis=1) #実測値との比較のため元データに予測した結果を結合
    
    # 多変量連関図（色分け）の描画
    sns.pairplot(df4, hue="Cluster No",diag_kind='hist',diag_kws={'alpha':0.3})
    
    return df4



####### 主成分分析 #######
def PrincipalComponentAnalysis_classification(df, explanatory_list):
    # Xに説明変数をセットする。
    X = df.loc[:,explanatory_list]
    
    # 説明変数を標準化する
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    # 主成分分析を行い各主成分(総合特性値)の割合を確認する
    mdl = PCA()
    X_std_pca = mdl.fit_transform(X_std)  # X_stdを対角化する
    mdl.explained_variance_ratio_
    
    # 主成分分析から算出された各主成分の割合をパレート図で確認する
    PC_list = []
    for i in range(len(explanatory_list)):
        PC_list.append('PC' + str(i+1))
    
    df2 = pd.concat([pd.DataFrame(PC_list,columns=['PCA No']),
                     pd.DataFrame(mdl.explained_variance_ratio_,columns=['PCA Value'])],axis=1)
    df2['accum'] = np.cumsum(df2['PCA Value'])
    df2["accum_percent"] = df2["accum"] / sum(df2["PCA Value"]) * 100
    
    # パレート図の描画
    fig, ax1 = plt.subplots(figsize=(10,7))
    data_num = len(df2)
    ax1.bar(range(data_num), df2["PCA Value"])
    ax1.set_xticks(range(data_num))
    ax1.set_xticklabels(df2["PCA No"].tolist())
    ax1.set_xlabel("Principal components")
    ax1.set_ylabel("Explained variance ratio")
    ax2 = ax1.twinx()
    ax2.plot(range(data_num), df2["accum_percent"], c="k", marker="o")
    ax2.set_ylim([0, 100])
    ax2.grid(True, which='both', axis='y')
    plt.show()
    
    # 主成分分析結果をグラフとして可視化し、データの特徴を確認
    plt.figure(figsize=(10, 7))
    plt.scatter(X_std_pca[:, 0], X_std_pca[:, 1],marker=".")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    pc0 = mdl.components_[0]
    pc1 = mdl.components_[1]
    for i in range(0, len(explanatory_list)):
        plt.arrow(0, 0, pc0[i]*5, pc1[i]*5, color='r')
        plt.text(pc0[i]*8, pc1[i]*5, X.columns[i], color='r')
    plt.show()


















