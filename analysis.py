import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import LineCollection
import seaborn as sns
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLasso
import scipy as sp
import networkx as nx



# 目的変数と他変数の関係図
def plot_target_other(df, n, t):
    fig = plt.figure(figsize = (15, 5*(n//3+1)))
    ax = [ ]
    for i in np.arange(1,n+1):
        ax_add = fig.add_subplot(n//3+1, 3, i)
        ax.append(ax_add)
    for i in np.arange(0, n):
        ax[i].scatter(df.iloc[:, t], df.iloc[:, i], label = df.columns[i])
        ax[i].set_xlabel(df.columns[t])
        ax[i].set_ylabel(df.columns[i])
        ax[i].grid(True)
        ax[i].legend()
    plt.show()



# 目的変数と他変数の関係図：マハラノビス距離の等高線あり
def plot_target_other_mahalanobis(df, n, t):
    fig = plt.figure(figsize = (15, 5*(n//3+1)))
    ax = [ ]
    for i in np.arange(1, n+1):
        ax_add = fig.add_subplot(n//3+1, 3, i)
        ax.append(ax_add)
    # 軸倍率の設定
    scale_min = 0.8
    scale_max = 1.2
    # グラフ描画
    for i in np.arange(0, n):
        # マハラノビス距離の計算：データ
        xy = pd.concat([pd.DataFrame(df.iloc[:, t]), pd.DataFrame(df.iloc[:, i])], axis=1)
        xy_cov = EmpiricalCovariance().fit(xy)
        xy_mahal = xy_cov.mahalanobis(xy)
        plt.xlim(xy.iloc[:, 0].min()*scale_min, xy.iloc[:, 0].max()*scale_max)
        plt.ylim(xy.iloc[:, 1].min()*scale_min, xy.iloc[:, 1].max()*scale_max)
        # マハラノビス距離の計算：メッシュグリッド
        xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 500),
                             np.linspace(plt.ylim()[0], plt.ylim()[1], 500))
        zz = np.c_[xx.ravel(), yy.ravel()]
        grid_mahal = xy_cov.mahalanobis(zz)
        grid_mahal_reshape = grid_mahal.reshape(xx.shape)
        # グラフ描画
        xy_mahal_mean = xy_mahal.mean()
        xy_mahal_std = xy_mahal.std()
        levels = [xy_mahal_mean, xy_mahal_mean+xy_mahal_std, xy_mahal_mean+2*xy_mahal_std, xy_mahal_mean+3*xy_mahal_std]
        ax[i].contour(xx, yy, grid_mahal_reshape, levels,cmap = plt.cm.jet, linestyles = 'solid') 
        ax[i].scatter(df.iloc[:,t], df.iloc[:,i], label = df.columns[i])
        ax[i].set_title(df.columns[i])
        ax[i].set_xlabel(df.columns[t])
        ax[i].set_ylabel(df.columns[i])
        ax[i].grid(True)
        ax[i].legend()
    plt.show()



# ヒストグラム
def plot_hist(df, n):
    fig = plt.figure(figsize = (15, 3*(n//3+1)))
    ax = [ ]
    for i in np.arange(1,n+1):
        ax_add = fig.add_subplot(n//3+1, 3, i)
        ax.append(ax_add)
    # グラフ引数の指定
    bins = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10] # デフォルト：10
    # グラフ描画
    for i in np.arange(0, n):
        ax[i].hist(df.iloc[:, i], bins=bins[i], label=df.columns[i])
        ax[i].set_title(df.columns[i])
        ax[i].grid(True)
        ax[i].legend()
    plt.show()



####### Graphical Lassoで相関分析 #######
def GraphicalLasso_correlation(df, explanatory_list):
    # 使う変数の列を抜き出す
    df1 = df.loc[:,explanatory_list]
    
    # 多変量連関図の描画
    sns.pairplot(df1)
    
    # 説明変数を標準化する
    sc = StandardScaler()
    X_std = sc.fit_transform(df1)
    
    # Graphical Lassoを用いて変数間の相関を求める。
    mdl = GraphicalLasso(alpha=0.1) # alpha は L１正則化パラメーター
    mdl.fit(X_std) #モデルにインプットデータXをあてはめる
    
    # 分散共分散行列（相関行列）の確認
    cov_ = mdl.covariance_
    
    # 相関行列のヒートマップ表示
    plt.figure(figsize=(15,7))
    label=list(df1.iloc[:,:].columns)
    sns.heatmap(mdl.covariance_,annot=True,fmt='0.2f',xticklabels=label, yticklabels=label,vmin=-1,vmax=1,center=0,square=True)
    plt.ylim(mdl.covariance_.shape[0],0)#表示サイズを合わせる
    plt.title("Graphical Lasso: Covariance matrix")
    
    # 偏相関行列の計算
    # 相関行列の逆行列から　要素ごとに -Rij/SQRT(Rii x Rjj)で計算している。
    X_cov_inv = sp.linalg.inv(cov_) #相関行列の逆行列(linalg.inv)
    # 偏相関係数
    pcm = np.empty_like(X_cov_inv)
    for i in range(X_cov_inv.shape[0]):
        for j in range(X_cov_inv.shape[0]):
            pcm[i, j] = -X_cov_inv[i, j]/np.sqrt(X_cov_inv[i, i]*X_cov_inv[j, j])
    
    ### 偏相関行列のヒートマップ表示
    plt.figure(figsize=(12,8))
    sns.heatmap(pcm,annot=True,fmt='0.2f',xticklabels=label, yticklabels=label,vmin=-1,vmax=1,center=0,square=True)
    plt.ylim(cov_.shape[0],0) #表示サイズを合わせる
    
    ### networkx ライブラリを使用したグラフ描画
    # 精度行列（偏相関行列）
    # ノード（頂点）の位置は描画毎に変化します
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Yu Gothic', 'Meirio']
    # フォントのサンプル　rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    
    threshold = 0.01 # これよりも弱いのは無視
    # 初期設定
    gr = nx.Graph()
    # ノードの情報を生成、ノードに重みはないので名前のみ
    gr.add_nodes_from(label)
    # labelからエッジ情報（辺）を生成
    edge_labels = {}
    for i, lbl1 in enumerate(label):
        for j, lbl2 in enumerate(label):
            if i >= j:
                continue
            if pcm[i,j] < threshold:
                continue
            gr.add_edge(lbl1, lbl2, weight=pcm[i,j])
            edge_labels[(lbl1,lbl2)] = str(round(pcm[i,j], 2))
    
    plt.figure(figsize=(15,12)) 
    pos = nx.spring_layout(gr, k=1.0)  # ノード間の反発力を定義、値が小さいほど密集（kが小さいほどゆがむ）
    nx.draw_networkx_nodes(gr, pos, node_color='b', alpha=0.2, font_weight="bold", font_family='VL Gothic') # ノードのスタイルを定義
    nx.draw_networkx_labels(gr, pos, font_size=16)
    edge_width = [d['weight']*10 for (u,v,d) in gr.edges(data=True)] # エッジ(辺)の太さを調整
    nx.draw_networkx_edges(gr, pos, alpha=0.4, edge_color='r', width=edge_width) # エッジのスタイルを定義
    nx.draw_networkx_edge_labels(gr, pos, edge_labels=edge_labels, font_color='red', font_size=16)
    plt.axis('off')
    plt.show()
    
    ### 位置を自分で制御（固定）したい場合の描画方法
    threshold = 0.01 # これよりも弱いのは無視
    size = len(label)
    # 位置を計算
    pos = np.zeros((size,2))
    for i in range(size):
        angle = (math.pi / 180.0) * ( 360.0 / size ) * i
        x = math.sin(angle) 
        y = -math.cos(angle)
        pos[i,0] = x
        pos[i,1] = y
    
    # 描画領域を設定
    plt.figure(figsize=(15,12))
    ax = plt.axes([-1.,-1.,1.,1.])
    # ノード名を表示
    for (xx,yy), l in zip(pos, label):
        plt.text(xx,yy, l, fontsize=20)
    # ラベル付与
    for i in range(len(label)):
        for j in range(len(label)):
            if i >= j :
                continue
            x = (pos[i][0]*0.8 + pos[j][0]*1.2) /2
            y = (pos[i][1]*0.8 + pos[j][1]*1.2) /2
            if pcm[i,j] >= threshold:
                val = round(pcm[i,j], 2)
                plt.text(x,y,str(val), fontsize=14)
    # segment作成(どこからどこに線を引くか)
    segments = []
    for i in range(size):
        for j in range(size):
            if pcm[i,j] >= threshold:
                segments += [[pos[i,:], pos[j,:]]]
    # 描画情報を設定(線の太さと色)
    widths = np.full(len(segments), 0.0)
    colors = []
    count = 0
    for i in range(size):
        for j in range(size):
            if pcm[i,j] >= threshold:
                widths[count] = pcm[i,j]*10+1
                count += 1
                colors += [plt.cm.Blues((pcm[i,j]+1)/2)]
    # 描画
    lc = LineCollection(segments, zorder=0, color=colors, norm=plt.Normalize(0,0.5))
    lc.set_linewidths(widths)
    ax.add_collection(lc)
    plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', lw=0, label='MDS')
    plt.axis("off")
    plt.show()










