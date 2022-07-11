import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import mglearn
import pydotplus
import io
from IPython.display import Image
import lightgbm as lgb



####### 重回帰分析 #######
def Multiple_Regression(X, Y):
    # 各変数を標準化（各説明変数を平均0、分散1に変換）
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    Y_std = sc.fit_transform(Y.values.reshape(-1,1))
    
    # 重回帰の実行
    mdl = LinearRegression()
    mdl.fit(X_std,Y_std)
    
    # 重回帰結果の表示
    decimal_p = 3 #小数点以下桁
    print(X.columns)
    print('偏回帰係数：',mdl.coef_.round(decimal_p))
    print('定数項：',mdl.intercept_.round(decimal_p))
    print('寄与率：',mdl.score(X_std,Y_std).round(decimal_p))
    
    # 予測
    Y_pred = mdl.predict(X_std)
    r2 = r2_score(Y_std, Y_pred) # 実測値と予測値のスコア算出
    print('r2スコア：', r2)
    
    # 標準化されていたmpgを元のスケールに戻す
    Y_std_inverse = sc.inverse_transform(Y_std)
    Y_pred_inverse = sc.inverse_transform(Y_pred)
    
    # 実測-予測グラフの描画
    plt.figure(figsize=(10,10))
    plt.scatter(Y_std_inverse, Y_pred_inverse, c='steelblue', edgecolor='white', s=70)
    plt.plot(Y_std_inverse,Y_std_inverse) # 直線描画のためY_std_inverseを利用
    plt.title("実測-予測グラフ", fontname="MS Gothic")
    plt.xlabel("実測値", fontname="MS Gothic")
    plt.ylabel("予測値", fontname="MS Gothic")
    plt.grid(True)
    plt.show()



####### 正則化（Elasticnet）回帰 #######
def Elastic_Net(X, Y):
    # ワンホットエンコーディング
    # X = pd.get_dummies(X, dummy_na=True, columns=['xx'])
    
    # 変数を標準化（各説明変数を平均0、分散1に変換）
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    Y_std = sc.fit_transform(Y.values.reshape(-1,1))
    
    # 学習データとテストデータを分ける
    X_train, X_test, Y_train, Y_test=train_test_split(X_std, Y_std, test_size=0.3, random_state=77)
    
    ### Elasticnetで予測
    decimal_p = 3 #小数点以下の桁数
    mdl = ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=100000, tol=0.01)
    mdl.fit(X_train, Y_train)
    print('定数項：',mdl.intercept_.round(decimal_p))
    print('寄与率：', mdl.score(X_train,Y_train))
    pd.DataFrame(mdl.coef_.round(decimal_p),index=X.columns, columns=['偏回帰係数'])
    
    # 汎化性能の検証
    Y_pred = mdl.predict(X_test)
    r2 = r2_score(Y_test, Y_pred) # 真値と予測値のスコア算出
    print('r2スコア：', r2)
    
    # 標準化されていたmpgを元のスケールに戻す
    Y_test_inverse = sc.inverse_transform(Y_test)
    Y_pred_inverse = sc.inverse_transform(Y_pred)
    
    # 実測-予測グラフの描画
    plt.figure(figsize=(10,10))
    plt.scatter(Y_test_inverse, Y_pred_inverse, c='steelblue', edgecolor='white', s=70)
    plt.plot(Y_test_inverse,Y_test_inverse) # 直線描画のためY_std_inverseを利用
    plt.title("実測-予測グラフ")
    plt.xlabel("実測値")
    plt.ylabel("予測値")
    plt.grid(True)
    
    # 予測したデータの確認
    df_Y_test_inverse = pd.DataFrame(Y_test_inverse,columns=['True'])
    df_Y_pred_inverse = pd.DataFrame(Y_pred_inverse,columns=['Pred']) 
    
    # それぞれを結合して比較
    prediction = pd.concat([df_Y_test_inverse,df_Y_pred_inverse],axis=1)
    
    return prediction



####### 線形判別分析 #######
def Linear_Discriminant_Analysis(X, Y_C, target_name, df):
    # 教師データとテストデータを 7:3 の割合で分離する
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_C, test_size=0.3, random_state=77)
    
    # 説明変数を標準化する
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # LDA(Linear Discriminant Analysis)
    mdl = LinearDiscriminantAnalysis()
    mdl.fit(X_train_std,Y_train)
    
    #学習データを分類予測する
    Y_train_pred = mdl.predict(X_train_std)
    
    df_Train = X_train.join([df.loc[:,[target_name]], Y_train])
    df_Train = df_Train.reset_index(drop=True)
    df_Train = pd.concat([df_Train, pd.DataFrame(Y_train_pred,columns=['Pred train'])],axis=1)
    
    # 学習データの混同行列
    cm_train_1=confusion_matrix(Y_train,Y_train_pred)
    print('混同行列 0:Negative,1:Positive')
    print('[ [TN,FP]')
    print('  [FN,TP] ]')
    print('\n混同行列：学習データ')
    print(cm_train_1)
    # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
    print('\nAccuracy(正解率)：学習データ　= ',f'{accuracy_score(Y_train, Y_train_pred):.03f}')
    print('Recall(再現率)：学習データ　= ',f'{recall_score(Y_train, Y_train_pred, pos_label=1):.03f}')
    print('Precision(適合率)：学習データ　= ',f'{precision_score(Y_train, Y_train_pred, pos_label=1):.03f}')
    print('F_measure(F値)：学習データ　= ',f'{f1_score(Y_train, Y_train_pred, pos_label=1):.03f}')
    
    #テストデータを分類予測する
    Y_test_pred = mdl.predict(X_test_std)
    
    df_Test = X_test.join([df.loc[:,[target_name]], Y_test])
    df_Test = df_Test.reset_index(drop=True)
    df_Test = pd.concat([df_Test, pd.DataFrame(Y_test_pred,columns=['Pred test'])],axis=1)
    
    # テストデータの混同行列
    cm_test_1=confusion_matrix(Y_test,Y_test_pred)
    print('混同行列 0:Negative,1:Positive')
    print('[ [TN,FP]')
    print('  [FN,TP] ]')
    print('\n混同行列：テストデータ')
    print(cm_test_1)
    # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
    print('\nAccuracy(正解率)：テストデータ　= ',f'{accuracy_score(Y_test, Y_test_pred):.03f}')
    print('Recall(再現率)：テストデータ　= ',f'{recall_score(Y_test, Y_test_pred, pos_label=1):.03f}')
    print('Precision(適合率)：テストデータ　= ',f'{precision_score(Y_test, Y_test_pred, pos_label=1):.03f}')
    print('F_measure(F値)：テストデータ　= ',f'{f1_score(Y_test, Y_test_pred, pos_label=1):.03f}')
    
    return df_Train, df_Test



####### SVM #######
def Support_Vector_Machine(X, Y_C, target_name, explanatory_list, df, kernel):
    # 教師データとテストデータを 7:3 の割合で分離する
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_C, test_size=0.3, random_state=77)
    
    # 説明変数を標準化する
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    
    ### 線形カーネルを用いたSVM
    if kernel == 'linear':
        # グリッドサーチするためのパラメーターを設定する
        param_grid = {'kernel':['linear'],
                      'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]}
        
        # チューニング：グリッドサーチで最適なハイパーパラメータを探査
        mdl = GridSearchCV(SVC(), param_grid, cv=10) # cvはk分割交差検証の分割数
        mdl.fit(X_train_std, Y_train) # 学習の実行
        
        # グリッドサーチのベストな結果を表示する
        mdl.best_estimator_.get_params()
        
        # グリッドサーチによるスコア結果
        grid_scores = pd.DataFrame(mdl.cv_results_)
        scores = np.array(grid_scores.mean_test_score).reshape(len(param_grid['C']),1)
        
        # グリッドサーチによるスコアを視える化
        plt.figure(figsize=(len(param_grid['C']),1))
        scores = scores.T # 描画の為に転置
        mglearn.tools.heatmap(scores, xlabel='C', xticklabels=param_grid['C'],ylabel='',yticklabels=param_grid['kernel'],fmt='%0.4f')
        
        #学習データを分類予測する
        Y_train_pred = mdl.predict(X_train_std)
        
        df_Train = X_train.join([df.loc[:,[target_name]], Y_train])
        df_Train = df_Train.reset_index(drop=True)
        df_Train = pd.concat([df_Train, pd.DataFrame(Y_train_pred,columns=['Pred train'])],axis=1)
    
        # 学習データの混同行列
        cm_train_1=confusion_matrix(Y_train,Y_train_pred)
        print('混同行列 0:Negative,1:Positive')
        print('[ [TN,FP]')
        print('  [FN,TP] ]')
        print('\n混同行列：学習データ')
        print(cm_train_1)
        # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
        print('\nAccuracy(正解率)：学習データ　= ',f'{accuracy_score(Y_train, Y_train_pred):.03f}')
        print('Recall(再現率)：学習データ　= ',f'{recall_score(Y_train, Y_train_pred, pos_label=1):.03f}')
        print('Precision(適合率)：学習データ　= ',f'{precision_score(Y_train, Y_train_pred, pos_label=1):.03f}')
        print('F_measure(F値)：学習データ　= ',f'{f1_score(Y_train, Y_train_pred, pos_label=1):.03f}')
        
        # 偏回帰係数&切片
        print('\n偏回帰係数')
        for i in range(len(explanatory_list)):
            print(explanatory_list[i] + ': ',mdl.best_estimator_.coef_[0,i])
        print('切片')
        print(mdl.best_estimator_.intercept_[0])
        
        # テストデータを分類予測する
        Y_test_pred = mdl.predict(X_test_std)
        
        df_Test = X_test.join([df.loc[:,[target_name]], Y_test])
        df_Test = df_Test.reset_index(drop=True)
        df_Test = pd.concat([df_Test, pd.DataFrame(Y_test_pred,columns=['Pred test'])],axis=1)
    
        # テストデータの混同行列
        cm_test_1=confusion_matrix(Y_test,Y_test_pred)
        print('混同行列 0:Negative,1:Positive')
        print('[ [TN,FP]')
        print('  [FN,TP] ]')
        print('\n混同行列：テストデータ')
        print(cm_test_1)
        # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
        print('\nAccuracy(正解率)：テストデータ　= ',f'{accuracy_score(Y_test, Y_test_pred):.03f}')
        print('Recall(再現率)：テストデータ　= ',f'{recall_score(Y_test, Y_test_pred, pos_label=1):.03f}')
        print('Precision(適合率)：テストデータ　= ',f'{precision_score(Y_test, Y_test_pred, pos_label=1):.03f}')
        print('F_measure(F値)：テストデータ　= ',f'{f1_score(Y_test, Y_test_pred, pos_label=1):.03f}')
    
    ### ガウスカーネルを用いたSVM
    elif kernel == 'rbf':
        # グリッドサーチするためのパラメーターを設定する
        param_grid = {'kernel':['rbf'],
                      'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                      'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]} # グリッドサーチするためのパラメータ
        
        # チューニング：グリッドサーチで最適なハイパーパラメータを探査
        mdl = GridSearchCV(SVC(), param_grid, cv=10) # cvはk分割交差検証の分割数
        mdl.fit(X_train_std, Y_train) # 学習の実行
        
        # グリッドサーチのベストな結果を表示する
        mdl.best_estimator_.get_params()
        
        # グリッドサーチによるスコア結果
        grid_scores = pd.DataFrame(mdl.cv_results_)
        scores = np.array(grid_scores.mean_test_score).reshape(len(param_grid['C']),len(param_grid['gamma']))
        
        # グリッドサーチによるスコアを視える化
        plt.figure(figsize=(len(param_grid['C']),len(param_grid['gamma'])))
        scores = scores.T # 描画の為に転置
        mglearn.tools.heatmap(scores, xlabel='C', xticklabels=param_grid['C'],ylabel='gamma',yticklabels=param_grid['gamma'],fmt='%0.4f')
        
        # 学習データを分類予測する
        Y_train_pred = mdl.predict(X_train_std)
        
        df_Train = X_train.join([df.loc[:,[target_name]], Y_train])
        df_Train = df_Train.reset_index(drop=True)
        df_Train = pd.concat([df_Train, pd.DataFrame(Y_train_pred,columns=['Pred train'])],axis=1)
        
        # 学習データの混同行列
        cm_train_1=confusion_matrix(Y_train,Y_train_pred)
        print('混同行列 0:Negative,1:Positive')
        print('[ [TN,FP]')
        print('  [FN,TP] ]')
        print('\n混同行列：学習データ')
        print(cm_train_1)
        # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
        print('\nAccuracy(正解率)：学習データ　= ',f'{accuracy_score(Y_train, Y_train_pred):.03f}')
        print('Recall(再現率)：学習データ　= ',f'{recall_score(Y_train, Y_train_pred, pos_label=1):.03f}')
        print('Precision(適合率)：学習データ　= ',f'{precision_score(Y_train, Y_train_pred, pos_label=1):.03f}')
        print('F_measure(F値)：学習データ　= ',f'{f1_score(Y_train, Y_train_pred, pos_label=1):.03f}')
        
        # テストデータを分類予測する
        Y_test_pred = mdl.predict(X_test_std)
        
        df_Test = X_test.join([df.loc[:,[target_name]], Y_test])
        df_Test = df_Test.reset_index(drop=True)
        df_Test = pd.concat([df_Test, pd.DataFrame(Y_test_pred,columns=['Pred test'])],axis=1)
    
        # テストデータの混同行列
        cm_test_1=confusion_matrix(Y_test,Y_test_pred)
        print('混同行列 0:Negative,1:Positive')
        print('[ [TN,FP]')
        print('  [FN,TP] ]')
        print('\n混同行列：テストデータ')
        print(cm_test_1)
        # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
        print('\nAccuracy(正解率)：テストデータ　= ',f'{accuracy_score(Y_test, Y_test_pred):.03f}')
        print('Recall(再現率)：テストデータ　= ',f'{recall_score(Y_test, Y_test_pred, pos_label=1):.03f}')
        print('Precision(適合率)：テストデータ　= ',f'{precision_score(Y_test, Y_test_pred, pos_label=1):.03f}')
        print('F_measure(F値)：テストデータ　= ',f'{f1_score(Y_test, Y_test_pred, pos_label=1):.03f}')

    return df_Train, df_Test



####### 決定木 #######
def Decision_Tree(X, Y_C, target_name, df):
    # 教師データとテストデータを 7:3 の割合で分離する
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_C, test_size=0.3, random_state=77)
    
    # ハイパーパラメータチューニング
    param_grid = {'max_leaf_nodes':np.arange(2, 10),
                  'max_depth':np.arange(2, 10),
                  'min_samples_leaf':np.arange(3, 10),
                  'min_impurity_decrease':np.logspace(-5, -0.5, 10)}
    
    # チューニング：グリッドサーチで探査
    mdl_2 = GridSearchCV(DecisionTreeClassifier(random_state = 77),
                         param_grid,
                         cv=10,
                         scoring='accuracy')
    mdl_2.fit(X_train, Y_train)
    print(mdl_2.__class__.__name__)
    print("最適なパラメーター =", mdl_2.best_params_, "Accuracy(正解率) =", f'{mdl_2.best_score_:.03f}')
    
    # グリッドサーチの結果
    df_GS_result = pd.DataFrame(mdl_2.cv_results_)
    
    # 作成した決定木モデル
    dot_data = io.StringIO()
    tree.export_graphviz(mdl_2.best_estimator_, out_file = dot_data, feature_names = X_train.columns, class_names = ['1', '0'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    
    # 説明変数の重要度
    print("説明変数の重要度")
    feature_importance = pd.DataFrame(mdl_2.best_estimator_.feature_importances_)
    feature_importance.index = X_train.columns
    feature_importance.columns = ['feature importance']
    print(feature_importance)
    
    # 学習データの分類予測
    Y_train_pred = mdl_2.best_estimator_.predict(X_train)
    
    df_Train = X_train.join([df.loc[:,[target_name]], Y_train])
    df_Train = df_Train.reset_index(drop=True)
    df_Train = pd.concat([df_Train, pd.DataFrame(Y_train_pred,columns=['Pred train'])],axis=1)
            
    # 学習データの混同行列
    cm_train_2 = confusion_matrix(Y_train, Y_train_pred)
    print('混同行列 0:Negative, 1:Positive')
    print('[ [TN,FP]')
    print('  [FN,TP] ]')
    print('\n混同行列：学習データ')
    print(cm_train_2)
    # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
    print('\nAccuracy(正解率)：学習データ　= ',f'{accuracy_score(Y_train, Y_train_pred):.03f}')
    print('Recall(再現率)：学習データ　= ',f'{recall_score(Y_train, Y_train_pred, pos_label=1):.03f}')
    print('Precision(適合率)：学習データ　= ',f'{precision_score(Y_train, Y_train_pred, pos_label=1):.03f}')
    print('F_measure(F値)：学習データ　= ',f'{f1_score(Y_train, Y_train_pred, pos_label=1):.03f}')
    
    # テストデータの分類予測
    Y_test_pred = mdl_2.best_estimator_.predict(X_test)
    
    df_Test = X_test.join([df.loc[:,[target_name]], Y_test])
    df_Test = df_Test.reset_index(drop=True)
    df_Test = pd.concat([df_Test, pd.DataFrame(Y_test_pred,columns=['Pred test'])],axis=1)
            
    # テストデータの混同行列
    cm_test_2 = confusion_matrix(Y_test, Y_test_pred)
    print('混同行列 0:Negative, 1:Positive')
    print('[ [TN,FP]')
    print('  [FN,TP] ]')
    print('\n混同行列：テストデータ')
    print(cm_test_2)
    # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
    print('\nAccuracy(正解率)：テストデータ　= ',f'{accuracy_score(Y_test, Y_test_pred):.03f}')
    print('Recall(再現率)：テストデータ　= ',f'{recall_score(Y_test, Y_test_pred, pos_label=1):.03f}')
    print('Precision(適合率)：テストデータ　= ',f'{precision_score(Y_test, Y_test_pred, pos_label=1):.03f}')
    print('F_measure(F値)：テストデータ　= ',f'{f1_score(Y_test, Y_test_pred, pos_label=1):.03f}')
    
    return df_GS_result, df_Train, df_Test



####### ランダムフォレスト #######
def Random_Forest(X, Y_C, target_name, df):
    # 教師データとテストデータを 7:3 の割合で分離する
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_C, test_size=0.3, random_state=77)
    
    # ハイパーパラメータチューニング
    param_grid = {'n_estimators':np.arange(95, 105),
                  'max_features':np.arange(1, 5),
                  'max_depth':np.arange(7, 12)}
    
    # チューニング：グリッドサーチで探査
    mdl_1 = GridSearchCV(RandomForestClassifier(oob_score = True, random_state = 77),
                         param_grid,
                         cv=10,
                         scoring='accuracy')
    mdl_1.fit(X_train, np.array(Y_train).reshape(-1))
    print(mdl_1.__class__.__name__)
    print("最適なパラメーター =", mdl_1.best_params_ , "，Accuracy(正解率) =", f'{mdl_1.best_score_:.03f}')
    
    # グリッドサーチの結果
    df_GS_result = pd.DataFrame(mdl_1.cv_results_)
    
    # 説明変数の重要度
    # 局所的な影響に対する感度が高い（局所的にはデータの分類に影響するため）
    # カーディナリティが低い場合に過小評価してしまう（分岐の回数が少なくなるため）
    # 多重共線性がある特徴量は過小評価してしまう（重要度を奪い合う）
    print("\n説明変数の重要度")
    feature_importance = pd.DataFrame(mdl_1.best_estimator_.feature_importances_)
    feature_importance.index = X_train.columns
    feature_importance.columns = ['feature importance']
    print(feature_importance)
    
    # 学習データの分類予測
    Y_train_pred = mdl_1.best_estimator_.predict(X_train)
    
    df_Train = X_train.join([df.loc[:,[target_name]], Y_train])
    df_Train = df_Train.reset_index(drop=True)
    df_Train = pd.concat([df_Train, pd.DataFrame(Y_train_pred,columns=['Pred train'])],axis=1)
    
    # 学習データの混同行列
    cm_train_1 = confusion_matrix(Y_train, Y_train_pred)
    print('混同行列 0:Negative, 1:Positive')
    print('[ [TN,FP]')
    print('  [FN,TP] ]')
    print('\n混同行列：学習データ')
    print(cm_train_1)
    # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
    print('\nAccuracy(正解率)：学習データ　= ',f'{accuracy_score(Y_train, Y_train_pred):.03f}')
    print('Recall(再現率)：学習データ　= ',f'{recall_score(Y_train, Y_train_pred, pos_label=1):.03f}')
    print('Precision(適合率)：学習データ　= ',f'{precision_score(Y_train, Y_train_pred, pos_label=1):.03f}')
    print('F_measure(F値)：学習データ　= ',f'{f1_score(Y_train, Y_train_pred, pos_label=1):.03f}')
    
    # OOB（Out-Of-Bag）データの分類結果
    oob_result = np.argmax(mdl_1.best_estimator_.oob_decision_function_, axis=1)
    
    df_OOB = X_train.join([df.loc[:,[target_name]], Y_train])
    df_OOB = df_Train.reset_index(drop=True)
    df_OOB = pd.concat([df_Train, pd.DataFrame(oob_result,columns=['Out-Of-Bag'])],axis=1)
    
    # OOBデータの混同行列
    cm_oob_1 = confusion_matrix(Y_train, oob_result)
    print('混同行列 0:Negative, 1:Positive')
    print('[ [TN,FP]')
    print('  [FN,TP] ]')
    print('\n混同行列：OOBデータ')
    print(cm_oob_1)
    # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
    print('\nAccuracy(正解率)：OOBデータ　= ',f'{accuracy_score(Y_train, oob_result):.03f}')
    print('Recall(再現率)：OOBデータ　= ',f'{recall_score(Y_train, oob_result, pos_label=1):.03f}')
    print('Precision(適合率)：OOBデータ　= ',f'{precision_score(Y_train, oob_result, pos_label=1):.03f}')
    print('F_measure(F値)：OOBデータ　= ',f'{f1_score(Y_train, oob_result, pos_label=1):.03f}')
    
    # テストデータの分類予測
    Y_test_pred = mdl_1.best_estimator_.predict(X_test)
    
    df_Test = X_test.join([df.loc[:,[target_name]], Y_test])
    df_Test = df_Test.reset_index(drop=True)
    df_Test = pd.concat([df_Test, pd.DataFrame(Y_test_pred,columns=['Pred test'])],axis=1)
    
    # テストデータの混同行列
    cm_test_1 = confusion_matrix(Y_test, Y_test_pred)
    print('混同行列 0:Negative, 1:Positive')
    print('[ [TN,FP]')
    print('  [FN,TP] ]')
    print('\n混同行列：テストデータ')
    print(cm_test_1)
    # 分類結果の評価：Accuracy(正解率),Recall(再現率),Precision(適合率),F_measure(F値)
    print('\nAccuracy(正解率)：テストデータ　= ',f'{accuracy_score(Y_test, Y_test_pred):.03f}')
    print('Recall(再現率)：テストデータ　= ',f'{recall_score(Y_test, Y_test_pred, pos_label=1):.03f}')
    print('Precision(適合率)：テストデータ　= ',f'{precision_score(Y_test, Y_test_pred, pos_label=1):.03f}')
    print('F_measure(F値)：テストデータ　= ',f'{f1_score(Y_test, Y_test_pred, pos_label=1):.03f}')
    
    return df_GS_result, df_Train, df_OOB, df_Test



### LightGBMで学習 ###
def Light_GBM(X_train, Y_train):
    rate_list = [0.05, 0.1, 0.2, 0.3, 0.4]
    depth_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    leaves_list = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    min_leaf_list = [1, 2, 3, 4, 5]

    # 学習回数
    esr = 300

    start_time = datetime.datetime.now()

    # トレーニングデータを学習用・検証用に分割
    X_trn, X_val, Y_trn, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

    # LightGBMの学習
    lgb_dataset_trn = lgb.Dataset(X_trn, label=Y_trn, categorical_feature='auto')
    lgb_dataset_val = lgb.Dataset(X_val, label=Y_val, categorical_feature='auto')

    # LightGBMのグリッドサーチ
    best_score = 0
    best_parameters = {}
    for rate in rate_list:
        for depth in depth_list:
            for leaves in leaves_list:
                for min_leaf in min_leaf_list:
                    params = {'objective' : 'multiclass',
                                'num_class' : 5, # 多クラスのクラス数を指定
                                # 'objective' : 'regression',
                                # 'objective' : 'binary', 
                                'metric': {'multi_error'},
                                # 'metric': {'mae'},
                                # 'metric': {'mse'},
                                # 'metric': {'binary_logloss'},
                                # 'metric': {'binary_error'}, # 評価指標 : 誤り率(= 1-正答率)
                                'early_stopping_rounds' : esr,   # early_stopping 回数指定
                                'learning_rate' : rate,
                                'max_depth' : depth,
                                'num_leaves': leaves,
                                'min_data_in_leaf': min_leaf
                                }

                    result_dic ={}
                    model = lgb.train(
                            params=params, 
                            train_set=lgb_dataset_trn, 
                            valid_sets=[lgb_dataset_trn, lgb_dataset_val], 
                            num_boost_round=10000, 
                            early_stopping_rounds=esr, 
                            # verbose_eval=100,
                            evals_result=result_dic
                            )

                    # train_pred = model.predict(X_train, num_iteration=model.best_iteration)
                    train_pred_prob = model.predict(X_train, num_iteration=model.best_iteration)
                    # train_pred = np.where(train_pred_prob < 0.5, 0, 1) # 0.5より小さい場合0 ,そうでない場合1を返す
                    train_pred = np.argmax(train_pred_prob, axis=1) # 最尤と判断したクラス
                    train_acc = accuracy_score(Y_train.values, train_pred)
                    # val_pred = model.predict(X_val, num_iteration=model.best_iteration)
                    val_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
                    # val_pred = np.where(val_pred_prob < 0.5, 0, 1)
                    val_pred = np.argmax(val_pred_prob, axis=1) # 最尤と判断したクラス
                    val_acc = accuracy_score(Y_val.values, val_pred)
                    print("rate  = ", rate)
                    print("depth = ", depth)
                    print("leaves = ", leaves)
                    print("min_leaf = ", min_leaf)
                    print(f'train acc : {train_acc:.3f}%')
                    print(f'valid acc : {val_acc:.3f}%')
                    
                    # 最も良いスコアのパラメータとスコアを更新
                    score = val_acc
                    if score > best_score:
                        best_score = score
                        best_parameters = {'rate' : rate,
                                            'depth' : depth,
                                            'leaves' : leaves,
                                            'min_leaf' : min_leaf}

    print('Best score: {}'.format(best_score))
    print('Best parameters: {}'.format(best_parameters))


    params = {'objective' : 'multiclass',
            'num_class' : 5, # 多クラスのクラス数を指定
            # 'objective' : 'regression',
            # 'objective' : 'binary', 
            'metric': {'multi_error'},
            # 'metric': {'mae'},
            # 'metric': {'mse'},
            # 'metric': {'binary_logloss'},
            # 'metric': {'binary_error'}, # 評価指標 : 誤り率(= 1-正答率)
            'early_stopping_rounds' : esr,   # early_stopping 回数指定
            'learning_rate' : best_parameters["rate"],
            'max_depth' : best_parameters["depth"],
            'num_leaves': best_parameters["leaves"],
            'min_data_in_leaf': best_parameters["min_leaf"]
            }

    result_dic ={}
    model = lgb.train(
            params=params, 
            train_set=lgb_dataset_trn, 
            valid_sets=[lgb_dataset_trn, lgb_dataset_val], 
            num_boost_round=10000, 
            early_stopping_rounds=esr, 
            # verbose_eval=100,
            evals_result=result_dic
            )

    # 学習経過を表示
    result_df = pd.DataFrame(result_dic['training']).add_prefix('train_').join(pd.DataFrame(result_dic['valid_1']).add_prefix('valid_'))
    fig, ax = plt.subplots(figsize=(11, 7))
    result_df[['train_multi_error', 'valid_multi_error']].plot(ax=ax)
    ax.set_ylabel('multi error')
    ax.set_xlabel('num of iteration')
    #ax.set_ylim(2, 8)
    ax.grid()

    # 特徴量の重要度出力
    feature_importance = pd.DataFrame({
                                        'feature_name' : model.feature_name(),
                                        'importance' : model.feature_importance(importance_type='gain'), 
                                        })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize = (16, 9))
    sns.set(font='Yu Gothic')
    sns.barplot(data=feature_importance, x='importance', y='feature_name')
    plt.savefig('feature_importance.png')

    calc_time = datetime.datetime.now() - start_time
    print(calc_time)






























