import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



# 時系列データの未来を予測するサンプル用データセットをflights_seabornから作成
def arrange_flights_seaborn(df):
    data = df[['passengers']].values

    seq_len = 36
    pred_len = 12
    batch_size = 1
    border1s = [0, 12 * 9 - seq_len, 12 * 11 - seq_len]
    border2s = [12 * 9, 12 * 11, 12 * 12]
    ss = StandardScaler()
    data = ss.fit_transform(data)

    train_data = data[border1s[0]:border2s[0]]
    val_data = data[border1s[1]:border2s[1]]
    test_data = data[border1s[2]:border2s[2]]

    class AirPassengersDataset(Dataset):
        def __init__(self, data, seq_len, pred_len):
            #学習期間と予測期間の設定
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.data = data

        def __getitem__(self, index):
            #学習用の系列と予測用の系列を出力
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len

            src = self.data[s_begin:s_end]
            tgt = self.data[r_begin:r_end]

            return src, tgt
        
        def __len__(self):
            return len(self.data) - self.seq_len - self.pred_len + 1

    train_set = AirPassengersDataset(data=train_data, seq_len=seq_len, pred_len=pred_len)
    val_set = AirPassengersDataset(data=val_data, seq_len=seq_len, pred_len=pred_len)
    test_set = AirPassengersDataset(data=train_data, seq_len=seq_len, pred_len=pred_len)

    #データをバッチごとに分けて出力できるDataLoaderを使用
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader



# Transformerで時系列から時系列を予測するサンプル用データセットをflights_seabornから作成
def arrange_flights_seaborn_Transformer(df):
    data = df[['passengers']].values

    # 説明変数の次元拡張
    dlen = 144  #ノイズデータのデータ長
    mean = 0.0  #ノイズの平均値
    std  = 1.0  #ノイズの分散

    # 目的変数
    tgt  = []
    for i in range(100):
        y = np.random.normal(mean,std,dlen).reshape(-1,1)
        tgt.append(data + data * y)
    tgt = np.array(tgt)

    # 説明変数
    exp = []
    for t in tgt:
        e = t.reshape(-1,)
        y = np.random.normal(mean,std,dlen)
        t1 = e - e * y
        t2 = e + e * y
        t3 = e + e * y * 2
        exp.append(np.array([t1, t2, t3]).T)
    exp = np.array(exp)

    for i in range(10):
        fig = plt.figure(figsize=(16,9))
        plt.rcParams['font.size'] = 12
        x = list(range(df[['month']].values.shape[0]))
        plt.plot(x,  tgt[i])
        plt.plot(x,  exp[i,:,0])
        plt.plot(x,  exp[i,:,1])
        plt.plot(x,  exp[i,:,2])
        plt.grid(True)
        plt.close()
        fig.savefig('result/data' + str(i+1) + '.png')

    X_, X_test, Y_, Y_test = train_test_split(exp, tgt, test_size=0.1, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_, Y_, test_size=0.2, random_state=0)


    class TimeSeriesDataset(Dataset):
        def __init__(self, exp, tgt):
            self.data = np.concatenate([exp, tgt], 2)
            self.exp = exp
            self.tgt = tgt

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.exp[idx], self.tgt[idx]
            # return torch.tensor(self.data[idx], dtype=torch.float32)

    batch_size = 16
    train_dataset = TimeSeriesDataset(X_train, Y_train)
    val_dataset = TimeSeriesDataset(X_val, Y_val)
    test_dataset = TimeSeriesDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

