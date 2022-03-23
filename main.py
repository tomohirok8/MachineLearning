import os
os.chdir('D:\\GitHub\\DS3')
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
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
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

from sub import check_pytorch
from data_read import My_Data_Read
from preprocessing import missing_value_variable, missing_value_sample, drop_missing, fill_missing, str_to_float,\
                            str_to_numeric
from analysis import plot_target_other, plot_target_other_mahalanobis, plot_hist, GraphicalLasso_correlation
from outlier import outlier_MT, outlier_OCSVM1, outlier_OCSVM2
from classification import hierarchical_cluster_analysis, kmeans_classification, GaussianMixtureModel_classification,\
                            PrincipalComponentAnalysis_classification
from MachineLearning import Multiple_Regression, Elastic_Net, Linear_Discriminant_Analysis,Support_Vector_Machine,\
                            Decision_Tree, Random_Forest



# Pytorch環境を確認
check_pytorch()

# データ表示数のセッティング
pd.set_option('display.max_columns', None) # 全列表示されるようにPandasの設定を変更する
pd.set_option('display.max_rows', None) # 全行表示されるようにPandasの設定を変更す

# データ読み込み
train_data, test_data = My_Data_Read.RedWineQuality()

# 列名取得
columns_list = train_data.columns

# 目的変数列名指定
target_name = 'quality'



##################### 前処理 #####################
# 読込データ情報の確認
train_data.info()
test_data.info()

### 欠損値処理 ###
# 欠損値の確認（変数軸）
missing_variables_train = missing_value_variable(train_data)
missing_variables_test = missing_value_variable(test_data)

# 欠損値の確認（サンプル軸）
missing_value_sample(train_data)
missing_value_sample(test_data)

# 欠損値の処理：データが閾値以上ある変数のみ残す＝欠損率（100%ー閾値）以上の変数を削除
drop_missing_thresh = 70
train_data_1 = drop_missing(train_data, drop_missing_thresh)
test_data_1 = drop_missing(test_data, drop_missing_thresh)

# 欠損のある行を埋める（変数軸）
# mean    : 平均値で埋める
# median  : 中央値で埋める
# unknown : unknownで埋める
# drop    : 欠損のある行を削除
method = 'mean'
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
train_data_1.info()
test_data_1.info()

# 数値なのに文字になっているデータの復元 カンマ除外,スペース除外,空白には0を入れ,?は0にする
str_train = []
str_test = []
str_to_float(train_data_1, str_train)
str_to_float(test_data_1, str_test)

# 数値の列すべてで数値以外のものを0に変更
keys_train = []
keys_test = []
str_to_numeric(train_data_1, keys_train)
str_to_numeric(test_data_1, keys_test)

# カテゴリ変数をダミー変数化する
dummy_train = []
dummy_test = []
for d in dummy_train:
    train_data_1 = pd.get_dummies(train_data_1, dummy_na=True, columns=[d])
for d in dummy_test:
    test_data_1 = pd.get_dummies(test_data_1, dummy_na=True, columns=[d])

### 最終回避処理 ###
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

# 変数の種類数
n_data = len(train_data_1.columns)
# 目的変数の列番号
target_num = list(train_data_1.columns).index(target_name)

# 目的変数と他変数の関係図
# マハラノビス距離の等高線有無設定
mahalanobis_use = 1
if mahalanobis_use == 1:
    plot_target_other_mahalanobis(train_data_1, n_data, target_num)
else:
    plot_target_other(train_data_1, n_data, target_num)

# ヒストグラム
plot_hist(train_data_1, n_data)

# 箱ひげ図
train_data_1.plot(kind='box', subplots=True, figsize=(15, 3*(n_data//3+1)), layout=(n_data//3+1, 3))
plt.show()



##################### MT法で外れ値検出 #####################
# 説明変数の列番号
explanatory_num = 5
# 2次元で外れ値検出
df_md = outlier_MT(train_data_1, columns_list, explanatory_num, target_num)



##################### 1クラスSVMで外れ値検出 #####################
# 使う変数の列名リストを作成
explanatory_num = 5
OCSVM_list = [columns_list[explanatory_num],columns_list[target_num]]
gamma_best = outlier_OCSVM1(train_data_1, OCSVM_list)
print(gamma_best)
outlier_OCSVM2(train_data_1, OCSVM_list, gamma_best)



##################### 階層的クラスター分析で層別 #####################
explanatory_list = columns_list[:11] # 説明変数名リスト
print(explanatory_list)
max_cluster = 5
df_HCA = hierarchical_cluster_analysis(train_data_1, explanatory_list, max_cluster)



##################### k-means法で層別 #####################
explanatory_list = columns_list[:11] # 説明変数名リスト
n_cluster = 5
df_kmeans = kmeans_classification(train_data_1, explanatory_list, n_cluster)



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


'''
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

'''

'''

##################### DNN #####################
# 説明変数名リスト
explanatory_list = columns_list[0:11]
# 目的変数名
target_name = columns_list[11]

x_train = train_data_1.loc[:,explanatory_list].values
t_train = train_data_1.loc[:,target_name].values
x_test = test_data_1.values

x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train):
        self.x_train = x_train
        self.t_train = t_train
        
    def __len__(self):
        return len(self.x_train)
        
    def __getitem__(self, idx):
        return torch.tensor(self.x_train[idx], dtype=torch.float), \
            torch.tensor(self.t_train[idx], dtype=torch.float)
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        self.x_test = x_test
        
    def __len__(self):
        return len(self.x_test)
        
    def __getitem__(self, idx):
        return torch.tensor(self.x_test[idx], dtype=torch.float)

train_dataset = TrainDataset(x_train, t_train)
valid_dataset = TrainDataset(x_valid, t_valid)
test_dataset = TestDataset(x_test)

class L1_penalty(nn.Module):
    def __init__(self, loss_fn, model, lambda_):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = model
        self.lambda_ = lambda_
    def __call__(self,pred,t,vis_penalty = False):
        loss_ = self.loss_fn(pred,t)
        # ノルム計算
        penalty = 0.0
        for param in self.model.parameters():
            penalty += torch.sum(torch.abs(param))
        penalty *= self.lambda_
        
        if(vis_penalty):
            print(penalty)
        return loss_ + penalty

class L2_penalty(nn.Module):
    def __init__(self, loss_fn, model, lambda_):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = model
        self.lambda_ = lambda_
    def __call__(self,pred,t,vis_penalty = False):
        loss_ = self.loss_fn(pred,t)
        # ノルム計算
        penalty = 0.0
        for param in self.model.parameters():
            penalty += torch.sum(param ** 2) ## L2正則化の場合の実装例
        penalty *= self.lambda_
        
        if(vis_penalty):
            print(penalty)
        return loss_ + penalty

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# DataLoaderの作成
BATCH_SIZE = 32
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

# ニューラルネットの定義
class MLP(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer_1 = nn.Linear(num_features, 32)  # 入力層
        self.layer_2 = nn.Linear(32, 256)  # 中間層
        self.layer_3 = nn.Linear(256, 256)  # 中間層
        self.layer_4 = nn.Linear(256, 16)  # 中間層
        self.layer_out = nn.Linear(16, 1)  # 出力層
        # self.bn1 = nn.BatchNorm1d(32)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(16)
        
    def forward(self, inputs):
        x = self.layer_1(inputs)
        # x = self.bn1(x)
        x = F.relu(x)
        # x = F.leaky_relu(x)
        # x = F.rrelu(x)
        # x = torch.tanh(x)
        x = F.dropout(x, p=0.5)#, training=self.training)
        x = self.layer_2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        # x = F.leaky_relu(x)
        # x = F.rrelu(x)
        # x = torch.tanh(x)
        x = F.dropout(x, p=0.5)#, training=self.training)
        x = self.layer_3(x)
        # x = self.bn2(x)
        x = F.relu(x)
        # x = F.leaky_relu(x)
        # x = F.rrelu(x)
        # x = torch.tanh(x)
        x = F.dropout(x, p=0.5)#, training=self.training)
        x = self.layer_4(x)
        # x = self.bn3(x)
        x = F.relu(x)
        # x = F.leaky_relu(x)
        # x = F.rrelu(x)
        # x = torch.tanh(x)
        x = self.layer_out(x)
        return x

NUM_FEATURES = 11
mlp = MLP(NUM_FEATURES)
mlp.to(DEVICE)

# 損失関数の定義
loss_fn = nn.MSELoss()
loss_fn = L1_penalty(loss_fn, mlp, 1e-4)
# loss_fn = L2_penalty(loss_fn, mlp, 1e-4)

# 学習率
LEARNING_RATE_list = [0.05]

# 学習の実行
# 精度向上ポイント: エポック数の大小
NUM_EPOCHS = 2000

earlystopping_rate_list = [100]

best_train_loss = 1000
best_valid_loss = 1000
best_list = [0,0,0,0]

for LEARNING_RATE in LEARNING_RATE_list:
    # オプティマイザの定義
    # optimizer = optim.Adam(mlp.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.AdamW(mlp.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.Adagrad(mlp.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.RMSprop(mlp.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.Adadelta(mlp.parameters(), lr=LEARNING_RATE)
    
    for earlystopping_rate in earlystopping_rate_list:
        earlystopping_count = 0
        print('LEARNING_RATE: ', LEARNING_RATE, ' earlystopping_rate: ', earlystopping_rate)
        
        loss_stats = {'train': [], 'valid': []}
        for e in range(1, NUM_EPOCHS+1):
            # 訓練
            train_epoch_loss = 0
            mlp.train()
            for x, t in train_loader:
                x, t = x.to(DEVICE), t.unsqueeze(1).to(DEVICE)
                optimizer.zero_grad()  # 勾配の初期化
                pred = mlp(x)  # 予測の計算(順伝播)
                loss = loss_fn(pred, t)  # 損失関数の計算
                loss.backward()  # 勾配の計算（逆伝播）
                optimizer.step()  # 重みの更新
                train_epoch_loss += loss.item()

            # 検証  
            with torch.no_grad():
                valid_epoch_loss = 0
                mlp.eval()
                for x, t in valid_loader:
                    x, t = x.to(DEVICE), t.unsqueeze(1).to(DEVICE)
                    pred = mlp(x)  # 予測の計算(順伝播)
                    loss = loss_fn(pred, t)  # 損失関数の計算
                    valid_epoch_loss += loss.item()

            loss_stats['train'].append(train_epoch_loss/len(train_loader))
            loss_stats['valid'].append(valid_epoch_loss/len(valid_loader))                              

            if e % 50 == 0 or e == NUM_EPOCHS:
                print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {valid_epoch_loss/len(valid_loader):.5f}')

            # ベストスコアの処理
            if valid_epoch_loss/len(valid_loader) < best_valid_loss:
                best_valid_loss = valid_epoch_loss/len(valid_loader)
                best_train_loss = train_epoch_loss/len(train_loader)
                best_list = [LEARNING_RATE, earlystopping_rate, best_train_loss, best_valid_loss]
                best_loss_stats = loss_stats
                # torch.save(mlp.state_dict(), '/root/userspace/best_model.pth')

            # Early Stoppingの処理
            if e == 1:
                earlystopping_score = valid_epoch_loss/len(valid_loader)

            if valid_epoch_loss/len(valid_loader) < earlystopping_score:
                earlystopping_score = valid_epoch_loss/len(valid_loader)
                earlystopping_count = 0
            else:
                earlystopping_score = earlystopping_score
                earlystopping_count += 1

            if earlystopping_count >= earlystopping_rate:
                print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {valid_epoch_loss/len(valid_loader):.5f} "Early Stopping"')
                break

print()
print(best_list)

# 推論の実行
mlp.eval()
preds = []
for x in test_loader:
    x = x.to(DEVICE)
    pred = mlp(x)
    pred = pred.squeeze()
    preds.extend(pred.tolist())

submission = pd.Series(preds, name='quality')
submission.to_csv('submission1_pred.csv', 
                  header=True, index_label='id')




'''