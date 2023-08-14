import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

import scipy as sp
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn import svm, mixture, tree
from sklearn.decomposition import PCA
from sklearn.covariance import GraphicalLasso
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
import itertools
import networkx as nx
import mglearn
from matplotlib import rcParams
from matplotlib.collections import LineCollection
import pydotplus
import io
from IPython.display import Image
import torch
from torch import nn, optim
import torch.nn.functional as F

os.chdir('D:\\GitHub\\DS3')
from code.sub import check_pytorch
from code.data_read import My_Data_Read
from code.preprocessing import missing_value_variable, missing_value_sample, drop_missing, fill_missing, str_to_float,\
                            str_to_numeric
from code.analysis import plot_target_other, plot_target_other_mahalanobis, plot_hist, GraphicalLasso_correlation
from code.outlier import outlier_MT, outlier_OCSVM1, outlier_OCSVM2
from code.classification import hierarchical_cluster_analysis, kmeans_classification, GaussianMixtureModel_classification,\
                            PrincipalComponentAnalysis_classification
from code.MachineLearning import Multiple_Regression, Elastic_Net, Linear_Discriminant_Analysis,Support_Vector_Machine,\
                            Decision_Tree, Random_Forest, Light_GBM



# Pytorch環境を確認
check_pytorch()

# データ表示数のセッティング
pd.set_option('display.max_columns', None) # 全列表示されるようにPandasの設定を変更する
pd.set_option('display.max_rows', None) # 全行表示されるようにPandasの設定を変更する



############## データを読み込んで変数名を取得 ##############
# データ読み込み
train_data, test_data = My_Data_Read.RedWineQuality()

# 列名取得
columns_list = list(train_data.columns)

# 目的変数列名指定
target_name = 'quality'

# 説明変数列名リスト
explanatory_list = list(train_data.columns)
explanatory_list.remove(target_name)




############## 実行内容読み込み ##############
df_exe = pd.read_csv('execution.csv', header=None, encoding='shift_jis')



############## 結果保存ディレクトリ ##############
savedir = 'result'
os.makedirs(savedir, exist_ok=True)



##################### 前処理 #####################
# 読込データ情報の確認
train_data.info()
test_data.info()

### 欠損値処理 ###
# 欠損値がある変数を確認
missing_variables_train = missing_value_variable(train_data)
missing_variables_test = missing_value_variable(test_data)

# データが閾値以上ある変数のみ残す＝欠損率（100%ー閾値）以上の変数を削除
drop_missing_thresh = int(df_exe.iat[2,1])
train_data_1 = drop_missing(train_data, drop_missing_thresh)
test_data_1 = drop_missing(test_data, drop_missing_thresh)

# 欠損のある行を埋める
# mean    : 平均値で埋める
# median  : 中央値で埋める
# unknown : unknownで埋める
# drop    : 欠損のある行を削除
method = df_exe.iat[3,1]
fill_method_train = []
for i in range(len(missing_variables_train)):
    fill_method_train.append(method)
fill_method_test = []
for i in range(len(missing_variables_test)):
    fill_method_test.append(method)
fill_missing(train_data_1, missing_variables_train, fill_method_train)
fill_missing(test_data_1, missing_variables_test, fill_method_test)
    
# 目的変数が欠損している行を削除
train_data_1.dropna(subset=[target_name], inplace=True)

### 文字列処理 ###
# 数値なのに文字になっているデータの復元 カンマ除外,スペース除外,空白には0を入れ,?は0にする
str_train = [] # 数値が文字になっている変数名リスト
str_test = [] # 数値が文字になっている変数名リスト
str_to_float(train_data_1, str_train)
str_to_float(test_data_1, str_test)

# 数値の列すべてで数値以外のものを0に変更
keys_train = [] # 数値の列にが文字が混入している変数名リスト
keys_test = [] # 数値の列にが文字が混入している変数名リスト
str_to_numeric(train_data_1, keys_train)
str_to_numeric(test_data_1, keys_test)

# カテゴリ変数をダミー変数化する
dummy_train = [] # ダミー変数化する変数名リスト
dummy_test = [] # ダミー変数化する変数名リスト
for d in dummy_train:
    train_data_1 = pd.get_dummies(train_data_1, dummy_na=True, columns=[d])
for d in dummy_test:
    test_data_1 = pd.get_dummies(test_data_1, dummy_na=True, columns=[d])

# 最後の手段（NaNを0埋め）
train_data_1 = train_data_1.fillna(0)
test_data_1 = test_data_1.fillna(0)



##################### 解析 #####################
# 各変数の要約統計量
train_data_1.describe()

# 多変量連関図
fig = sns.pairplot(train_data_1, height=3, aspect=16/9, plot_kws={'alpha':0.5})
plt.show()
fig.savefig(savedir + '/pairplot_train.png')

# 各変数間の相関係数
df_corr = train_data_1.corr()
df_corr.to_csv(savedir + '/corr_train.csv')
sns.heatmap(df_corr, vmax=1, vmin=-1, center=0)
plt.savefig(savedir + '/corr_heatmap_train.png')

# 目的変数と他変数の関係図
# 変数の種類数
n_data = len(train_data_1.columns)
# 目的変数の列番号
target_num = list(train_data_1.columns).index(target_name)

# マハラノビス距離の等高線有無設定
mahalanobis_use = 1
if mahalanobis_use == 1:
    plot_target_other_mahalanobis(train_data_1, n_data, target_num, savedir)
else:
    plot_target_other(train_data_1, n_data, target_num)

# ヒストグラム
plot_hist(train_data_1, n_data, savedir)

# 箱ひげ図
if df_exe.iat[9,1] == 1:
    train_data_1.plot(kind='box', subplots=True, figsize=(15,3*(n_data//3+1)), layout=(n_data//3+1, 3))
    plt.savefig(savedir + '/box_train.png')



##################### 外れ値検出 #####################
# 説明変数の列番号
explanatory_list = ['fixed acidity', 'density']
### MT法で外れ値検出 ###
if df_exe.iat[10,1] == 1:
    df_md_list = []
    for expl in explanatory_list:
        df_md = outlier_MT(train_data_1, expl, target_name, savedir)
        df_md_list.append(df_md)
    df_md_all = pd.concat(df_md_list, axis=1)
    df_md_all.to_csv(savedir + '/outlier_md_train.csv', index=False)

### 1クラスSVMで外れ値検出 ###
for expl in explanatory_list:
    OCSVM_list = [expl, target_name]
    gamma_best = outlier_OCSVM1(train_data_1, OCSVM_list)
    print(gamma_best)
    outlier_OCSVM2(train_data_1, OCSVM_list, gamma_best, savedir)



##################### 層別 #####################
#### 階層的クラスター分析 ####
explanatory_list = columns_list[:11] # 説明変数名リスト
print(explanatory_list)
max_cluster = 5
df_HCA = hierarchical_cluster_analysis(train_data_1, explanatory_list, max_cluster)

### k-means法で層別 ###
explanatory_list = columns_list[:11] # 説明変数名リスト
n_cluster = 5
df_kmeans = kmeans_classification(train_data_1, explanatory_list, n_cluster)

### 混合ガウス分布で層別 ###
explanatory_list = columns_list[:11] # 説明変数名リスト
df_GMM = GaussianMixtureModel_classification(train_data_1, explanatory_list)

### 主成分分析 ###
explanatory_list = columns_list[:11] # 説明変数名リスト
PrincipalComponentAnalysis_classification(train_data_1, explanatory_list)

### Graphical Lassoで相関分析 ###
explanatory_list = ['fixed acidity', 'volatile acidity', 'density', 'pH', 'alcohol', 'quality'] # 説明変数名リスト
GraphicalLasso_correlation(train_data_1, explanatory_list)



##################### 回帰 #####################
### 重回帰分析 ###
# 説明変数名リスト
explanatory_list = ['fixed acidity', 'volatile acidity', 'density', 'pH', 'alcohol']
# 説明変数と目的変数に分割
X = train_data_1.loc[:,explanatory_list]
Y = train_data_1.loc[:,target_name]
Multiple_Regression(X, Y)

### 正則化（Elasticnet）回帰 ###
# 説明変数名リスト
explanatory_list = ['fixed acidity', 'volatile acidity', 'density', 'pH', 'alcohol']
# 説明変数と目的変数に分割
X = train_data_1.loc[:,explanatory_list]
Y = train_data_1.loc[:,target_name]
prediction = Elastic_Net(X, Y)



##################### 機械学習 #####################
### 線形判別分析 ###
# 説明変数名リスト
explanatory_list = ['fixed acidity', 'volatile acidity', 'density', 'pH', 'alcohol']
# 目的変数名
target_name = 'quality'
# 説明変数と目的変数に分割
X = train_data_1.loc[:,explanatory_list]
Y = train_data_1.loc[:,target_name]
# Yをカテゴリ変数に置き換える
str_list = []
for s in Y:
    if s > 5 :
        str_list.append(1)
    else:
        str_list.append(0)
Y_C = pd.DataFrame(str_list, columns=[target_name + ' Category'])

df_LDA_train, df_LDA_test = Linear_Discriminant_Analysis(X, Y_C, target_name, train_data_1)

###SVM ###
# 説明変数名リスト
explanatory_list = ['fixed acidity', 'volatile acidity', 'density', 'pH', 'alcohol']
# 目的変数名
target_name = 'quality'
# 説明変数と目的変数に分割
X = train_data_1.loc[:,explanatory_list]
Y = train_data_1.loc[:,target_name]
# Yをカテゴリ変数に置き換える
str_list = []
for s in Y:
    if s > 5 :
        str_list.append(1)
    else:
        str_list.append(0)
Y_C = pd.DataFrame(str_list, columns=[target_name + ' Category'])

df_SVM_train, df_SVM_test = Support_Vector_Machine(X, Y_C, target_name, explanatory_list, train_data_1, 'rbf')

### 決定木 ###
# 説明変数名リスト
explanatory_list = ['fixed acidity', 'volatile acidity', 'density', 'pH', 'alcohol']
# 目的変数名
target_name = 'quality'
# 説明変数と目的変数に分割
X = train_data_1.loc[:,explanatory_list]
Y = train_data_1.loc[:,target_name]
# Yをカテゴリ変数に置き換える
str_list = []
for s in Y:
    if s > 5 :
        str_list.append(1)
    else:
        str_list.append(0)
Y_C = pd.DataFrame(str_list, columns=[target_name + ' Category'])

df_DT_GSresult, df_DT_train, df_DT_test = Decision_Tree(X, Y_C, target_name, train_data_1)

### ランダムフォレスト ###
# 説明変数名リスト
explanatory_list = ['fixed acidity', 'volatile acidity', 'density', 'pH', 'alcohol']
# 目的変数名
target_name = 'quality'
# 説明変数と目的変数に分割
X = train_data_1.loc[:,explanatory_list]
Y = train_data_1.loc[:,target_name]
# Yをカテゴリ変数に置き換える
str_list = []
for s in Y:
    if s > 5 :
        str_list.append(1)
    else:
        str_list.append(0)
Y_C = pd.DataFrame(str_list, columns=[target_name + ' Category'])

df_RF_GSresult, df_RF_train, df_RF_oob, df_RF_test = Random_Forest(X, Y_C, target_name, train_data_1)


### LightGBMで学習 ###
# 説明変数と目的変数に分割
X = train_data_1[explanatory_list]
Y = train_data_1[target_name]

df_lightGBM, best_parameters, feature_importance = Light_GBM(X, Y)



