from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from torch import optim
from typing import NewType
import os
os.chdir('D:/GitHub/DS3')
import matplotlib.pyplot as plt



####### 前処理 #######
# 入力画像をAlexNetに合わせてサイズが224の正方形に変換
# ImageNetによる学習済みモデルの設定値：RGBの各値の平均値と標準偏差
transform = transforms.Compose([
    transforms.Resize(256),  # 短辺256
    transforms.CenterCrop(224),  # 224×224の正方形を中央から切り抜き
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar10_train = CIFAR10("/data", train=True, download=True, transform=transform)
cifar10_test = CIFAR10("/data", train=False, download=True, transform=transform)

# DataLoaderの設定
batch_size = 64
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

ModelType = 'AlexNet'
'''
AlexNet
GoogLeNet
VGG
ResNet
DenseNet
MobileNet
VisionTransformer
'''


####### モデル読み込み #######
# pretrained=Trueと設定することで、パラメータが訓練済みとなります。
if ModelType == 'AlexNet':
    # AlexNet
    net = models.alexnet(pretrained=True)
elif ModelType == 'GoogLeNet':
    # GoogLeNet
    net = models.googlenet(pretrained=True, aux_logits=True, transform_input=False)
elif ModelType == 'VGG':
    # VGG
    net = models.vgg11(pretrained=True)
elif ModelType == 'ResNet':
    net = models.resnet18(pretrained=True)
elif ModelType == 'DenseNet':
    net = models.densenet121(pretrained=True)
elif ModelType == 'MobileNet':
    net = models.mobilenet_v2(pretrained=True)
elif ModelType == 'VisionTransformer':
    net = models.vit_b_16(pretrained=True)

# モデルを確認
print(net)


####### 各層の設定 #######
# 分類器として機能する全結合層を10クラス分類に合わせて入れ替え
# 特徴抽出に用いた箇所は追加で訓練を行わず、分類器のみ訓練
# 全ての層のパラメータを訓練不可に
for param in net.parameters():
    param.requires_grad = False

# 一部の層を入れ替え：最後が10クラスとなるように少しずつ層の大きさを減らす
if ModelType == 'AlexNet':
    net.classifier[1] = nn.Linear(9216,4096)
    net.classifier[4] = nn.Linear(4096,1024)
    net.classifier[6] = nn.Linear(1024,10)
elif ModelType == 'GoogLeNet':
    # 補助の分類器 (Auxiliary Classifier) の出力層を入れ替え
    net.aux1.fc2 = nn.Linear(1024,10)
    net.aux2.fc2 = nn.Linear(1024,10)
    # 最後の出力層を入れ替え
    net.fc = nn.Linear(1024,10)
elif ModelType == 'VGG':
    net.classifier[6] = nn.Linear(4096,10)
elif ModelType == 'ResNet':
    net.fc = nn.Linear(512, 10)
elif ModelType == 'DenseNet':
    net.classifier = nn.Linear(1024, 10)
elif ModelType == 'MobileNet':
    net.classifier[1] = nn.Linear(1280, 10)
elif ModelType == 'VisionTransformer':
    net.heads[0] = nn.Linear(768, 10)

# GPU対応
net.cuda()

# 入れ替え後のモデルを確認
print(net)


####### 学習 #######
# 交差エントロピー誤差関数
loss_fnc = nn.CrossEntropyLoss()

# 最適化アルゴリズム
if ModelType == 'VisionTransformer':
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
else:
    optimizer = optim.Adam(net.parameters())

# 損失のログ
record_loss_train = []
record_loss_test = []

# 学習
if ModelType == 'AlexNet' or ModelType == 'VGG' or ModelType == 'ResNet'\
      or ModelType == 'DenseNet' or ModelType == 'MobileNet':
    for i in range(6):
        net.train()
        loss_train = 0
        for j, (x, t) in enumerate(train_loader):
            x, t = x.cuda(), t.cuda()  # GPU対応
            y = net(x)
            loss = loss_fnc(y, t)
            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train /= j+1
        record_loss_train.append(loss_train)

        net.eval()
        loss_test = 0
        for j, (x, t) in enumerate(test_loader):
            x, t = x.cuda(), t.cuda()
            y = net(x)
            loss = loss_fnc(y, t)
            loss_test += loss.item()
        loss_test /= j+1
        record_loss_test.append(loss_test)

        if i%1 == 0:
            print("Epoch:", i, "Loss_Train:", loss_train, "Loss_Test:", loss_test)

elif ModelType == 'GoogLeNet':
    for i in range(6):
        net.train()
        loss_train = 0
        for j, (x, t) in enumerate(train_loader):
            x, t = x.cuda(), t.cuda()
            y = net(x)
            loss_main = loss_fnc(y[0], t)  # 最後の出力層の損失
            loss_aux1 = loss_fnc(y[1], t)  # 補助の分類器の損失
            loss_aux2 = loss_fnc(y[2], t)  # 補助の分類器の損失
            loss = loss_main + 0.3*loss_aux1 + 0.3*loss_aux2
            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train /= j+1
        record_loss_train.append(loss_train)

        net.eval()  # 評価モード
        loss_test = 0
        for j, (x, t) in enumerate(test_loader):  # ミニバッチ（x, t）を取り出す
            x, t = x.cuda(), t.cuda()
            y = net(x)
            loss = loss_fnc(y, t)
            loss_test += loss.item()
        loss_test /= j+1
        record_loss_test.append(loss_test)

        if i%1 == 0:
            print("Epoch:", i, "Loss_Train:", loss_train, "Loss_Test:", loss_test)

# 誤差の推移
plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

# 正解率
correct = 0
total = 0
net.eval()
for i, (x, t) in enumerate(test_loader):
    x, t = x.cuda(), t.cuda()
    y = net(x)
    correct += (y.argmax(1) == t).sum().item()
    total += len(x)
print("正解率:", str(correct/total*100) + "%")





