import pandas as pd
import math



# 欠損値の確認：変数軸
def missing_value_variable(df):
    null_column = pd.concat([pd.DataFrame(df.isnull().sum(), columns={'The Number of Null'}),
                            pd.DataFrame(df.isnull().sum()/df.shape[0]*100, columns={'Null Percentage'}),
                            pd.DataFrame(df.mean(), columns={'Mean'}),
                            pd.DataFrame(df.median(), columns={'Median'})
                            ],
                        axis=1, sort=False)
    print(null_column)
    
    missing_value = df.isnull().sum()
    variables = []
    for i in range(len(missing_value)):
        if missing_value[i] >0:
            variables.append(missing_value.index[i])
    
    return variables



# 欠損値の確認：サンプル軸
def missing_value_sample(df):
    null_row = pd.concat([pd.DataFrame(df.isnull().sum(axis = 1).value_counts(sort = False), columns = {'The Number of Row'}),
                        pd.DataFrame(df.isnull().sum(axis = 1).value_counts(sort = False)/df.shape[0]*100, columns = {'Row Percentage'})],
                        axis = 1)
    null_row.index.names = ['The Number of Null']
    null_row = null_row.reset_index()
    null_row = pd.concat([null_row,pd.DataFrame(null_row.iloc[:, 0]/df.shape[1]*100).rename(columns = {'The Number of Null':'Null Percentage'})],
                        axis = 1).iloc[:, [0, 3, 1, 2]]
    null_row.sort_values('The Number of Null')
    print(null_row)



# 欠損値の処理：データが閾値以上ある変数のみ残す＝欠損率（100%ー閾値）以上の変数を削除
def drop_missing(df, drop_thresh):
    df1 = df.dropna(thresh=math.ceil(df.shape[0]*drop_thresh/100), axis=1)
    
    return df1



# 欠損のある行を埋める
def fill_missing(df, variable, method):
    for v, m in zip(variable, method):
        if m == 'mean':
            df[v].fillna(df.mean()[v], inplace=True)
        if m == 'median':
            df[v].fillna(df.median()[v], inplace=True)
        if m == 'unknown':
            df[v].fillna('unknown', inplace=True)
        if m == 'drop':
            df.dropna(subset=[v], inplace=True)



# 数値なのに文字になっているデータの復元 カンマ除外,スペース除外,空白には0を入れる
def str_to_float(df, str_):
    for s in str_:
        df[s] = list(map(lambda x : float(str(x).replace(',', '').replace(' ', '').replace('?', str(0))), df[s]))



# 数値の列すべてで数値以外のものを0に変更
def str_to_numeric(df, keys):
    for key in keys:
        df[key] = pd.to_numeric(df[key], errors='coerce').fillna(0)












