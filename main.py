import os
os.chdir('D:\\GitHub\\DS3')
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
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
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import itertools
import networkx as nx
import mglearn
from matplotlib import rcParams
from matplotlib.collections import LineCollection
import pydotplus
import io
from IPython.display import Image

from data_read import My_Data_Read
from preprocessing import missing_value_variable, missing_value_sample, drop_missing, fill_missing, str_to_float,\
                          str_to_numeric
from analysis import plot_target_other, plot_target_other_mahalanobis, plot_hist, GraphicalLasso_correlation
from outlier import outlier_MT, outlier_OCSVM
from classification import hierarchical_cluster_analysis, kmeans_classification, GaussianMixtureModel_classification,\
                           PrincipalComponentAnalysis_classification
from MachineLearning import Multiple_Regression, Elastic_Net, Linear_Discriminant_Analysis,Support_Vector_Machine,\
                            Decision_Tree, Random_Forest



# データ表示数のセッティング
pd.set_option('display.max_columns', None) # 全列表示されるようにPandasの設定を変更する
pd.set_option('display.max_rows', None) # 全行表示されるようにPandasの設定を変更す

# データ読み込み
train_data, test_data = My_Data_Read.RedWineQuality()

# 列名取得
columns_list = train_data.columns

# 目的変数列名指定
target_name = 'quality'

# x_train = train_data[:, :-1]
# t_train = train_data[:, -1]

# train = train_data.fillna(train_data.median()).values
# x_test = test_data.fillna(test_data.median()).values


##################### 前処理 #####################
# 読込データ情報の確認
train_data.info()
test_data.info()

# 欠損値の確認（変数軸）
missing_variables_train = missing_value_variable(train_data)
missing_variables_test = missing_value_variable(test_data)

# 欠損値の確認（サンプル軸）
missing_value_sample(train_data)
missing_value_sample(test_data)

# 欠損値の処理：データが閾値以上ある変数のみ残す＝欠損率（100%ー閾値）以上の変数を削除
train_data_1 = drop_missing(train_data, 70)
test_data_1 = drop_missing(test_data, 70)

# 欠損のある行を埋める（変数軸）
# mean    : 平均値で埋める
# median  : 中央値で埋める
# unknown : unknownで埋める
# drop    : 欠損のある行を削除
fill_method_train = ['mean', 'mean']
fill_method_test = ['mean', 'mean']
fill_missing(train_data_1, missing_variables_train, fill_method_train)
fill_missing(test_data_1, missing_variables_test, fill_method_test)
    
# 目的変数が欠損している行を削除
train_data_1.dropna(subset=[target_name], inplace=True)

# 数値なのに文字になっているデータの復元 カンマ除外,スペース除外,空白には0を入れ,?は0にする
str_train = ['xx']
str_test = ['xx']
str_to_float(train_data_1, str_train)
str_to_float(test_data_1, str_test)

# 数値の列すべてで数値以外のものを0に変更
keys_train = ['xx']
keys_test = ['xx']
str_to_numeric(train_data_1, keys_train)
str_to_numeric(test_data_1, keys_test)

# カテゴリ変数をダミー変数化する
train_data_1 = pd.get_dummies(train_data_1, dummy_na=True, columns=['xx'])
test_data_1 = pd.get_dummies(test_data_1, dummy_na=True, columns=['xx'])

# これだけやってもまだ残っているNaNはとりあえず0で埋める
train_data_1 = train_data_1.fillna(0)
test_data_1 = test_data_1.fillna(0)



##################### 解析 #####################
# 各変数の要約統計量
train_data_1.describe()

# 多変量連関図
sns.pairplot(train_data_1, plot_kws = {'alpha':0.3})
plt.show()

# 各変数間の相関係数
train_data_1.corr()

# 変数の種類数と目的変数の列番号
n_data = len(train_data_1.columns)
target_num = 11 # 目的変数の列番号

# 目的変数と他変数の関係図
plot_target_other(train_data_1, n_data, target_num)
plot_target_other_mahalanobis(train_data_1, n_data, target_num) # マハラノビス距離の等高線あり

# ヒストグラム
plot_hist(train_data_1, n_data)

# 箱ひげ図
train_data_1.plot(kind='box', subplots=True, figsize=(15, 3*(n_data//3+1)), layout=(n_data//3+1, 3))
plt.show()



##################### MT法で外れ値検出 #####################
# 説明変数と目的変数の列番号
explanatory_num = 6
target_num = target_num
df_md = outlier_MT(train_data_1, columns_list, explanatory_num, target_num)



##################### 1クラスSVMで外れ値検出 #####################
# 使う変数の列名リストを作成
OCSVM_list = [columns_list[6],columns_list[11]]
outlier_OCSVM(train_data_1, OCSVM_list)



##################### 階層的クラスター分析で層別 #####################
explanatory_list = columns_list[:11] # 説明変数名リスト
df_HCA = hierarchical_cluster_analysis(train_data_1, explanatory_list, 5)



##################### k-means法で層別 #####################
explanatory_list = columns_list[:11] # 説明変数名リスト
df_kmeans = kmeans_classification(train_data_1, explanatory_list, 5)



##################### 混合ガウス分布で層別 #####################
explanatory_list = columns_list[:11] # 説明変数名リスト
df_GMM = GaussianMixtureModel_classification(train_data_1, explanatory_list)



##################### 主成分分析 #####################
explanatory_list = columns_list[:11] # 説明変数名リスト
PrincipalComponentAnalysis_classification(train_data_1, explanatory_list)



##################### Graphical Lassoで相関分析 #####################
explanatory_list = ['fixed acidity', 'volatile acidity', 'density', 'pH', 'alcohol', 'quality'] # 説明変数名リスト
GraphicalLasso_correlation(train_data_1, explanatory_list)



##################### 重回帰分析 #####################
# 説明変数名リスト
explanatory_list = ['fixed acidity', 'volatile acidity', 'density', 'pH', 'alcohol']
# 目的変数名
target_name = 'quality'
# 説明変数と目的変数に分割
X = train_data_1.loc[:,explanatory_list]
Y = train_data_1.loc[:,target_name]

Multiple_Regression(X, Y)


##################### 正則化（Elasticnet）回帰 #####################
# 説明変数名リスト
explanatory_list = ['fixed acidity', 'volatile acidity', 'density', 'pH', 'alcohol']
# 目的変数名
target_name = 'quality'
# 説明変数と目的変数に分割
X = train_data_1.loc[:,explanatory_list]
Y = train_data_1.loc[:,target_name]

prediction = Elastic_Net(X, Y)



##################### 線形判別分析 #####################
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



##################### SVM #####################
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



##################### 決定木 #####################
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



##################### ランダムフォレスト #####################
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





