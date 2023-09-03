############## 1.PyTorch-Transformersの基本 ##############
# PyTorch-Transformersのモデル
import torch
from pytorch_transformers import BertForMaskedLM

msk_model = BertForMaskedLM.from_pretrained('bert-base-uncased')  # 訓練済みパラメータの読み込み
print(msk_model)

from pytorch_transformers import BertForSequenceClassification

sc_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')  # 訓練済みパラメータの読み込み
print(sc_model)

# BertConfigクラスを使って、モデルの設定を行う
from pytorch_transformers import BertConfig

config = BertConfig.from_pretrained("bert-base-uncased")
print(config) 

# BertTokenizerクラスを使って、訓練済みのデータに基づく形態素解析を行う
from pytorch_transformers import BertTokenizer

text = "I have a pen. I have an apple."

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
words = tokenizer.tokenize(text)
print(words)



############## 2.シンプルなBERTの実装 ##############
import torch
from pytorch_transformers import BertForMaskedLM
from pytorch_transformers import BertTokenizer

### 文章の一部の予測 ###
text = "[CLS] I played baseball with my friends at school yesterday [SEP]"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
words = tokenizer.tokenize(text)
print(words)

# 文章の一部をMASK
msk_idx = 3
words[msk_idx] = "[MASK]"  # 単語を[MASK]に置き換える
print(words)

# 単語を対応するインデックスに変換
word_ids = tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
word_tensor = torch.tensor([word_ids])  # テンソルに変換
print(word_tensor)

# BERTのモデルを使って予測
msk_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
msk_model.cuda()  # GPU対応
msk_model.eval()

x = word_tensor.cuda()  # GPU対応
y = msk_model(x)  # 予測
result = y[0]
print(result.size())  # 結果の形状

_, max_ids = torch.topk(result[0][msk_idx], k=5)  # 最も大きい5つの値
result_words = tokenizer.convert_ids_to_tokens(max_ids.tolist())  # インデックスを単語に変換
print(result_words)

### 文章が連続しているかどうかの判定 ###
from pytorch_transformers import BertForNextSentencePrediction

def show_continuity(text, seg_ids):
    words = tokenizer.tokenize(text)
    word_ids = tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
    word_tensor = torch.tensor([word_ids])  # テンソルに変換

    seg_tensor = torch.tensor([seg_ids])

    nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    nsp_model.cuda()  # GPU対応
    nsp_model.eval()

    x = word_tensor.cuda()  # GPU対応
    s = seg_tensor.cuda()  # GPU対応

    y = nsp_model(x, s)  # 予測
    result = torch.softmax(y[0], dim=1)
    print(result)  # Softmaxで確率に
    print(str(result[0][0].item()*100) + "%の確率で連続しています。")

# show_continuity関数に、自然につながる2つの文章を与える
text = "[CLS] What is baseball ? [SEP] It is a game of hitting the ball with the bat [SEP]"
seg_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1]  # 0:前の文章の単語、1:後の文章の単語
show_continuity(text, seg_ids)

# show_continuity関数に、自然につながらない2つの文章を与える
text = "[CLS] What is baseball ? [SEP] This food is made with flour and milk [SEP]"
seg_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 0:前の文章の単語、1:後の文章の単語
show_continuity(text, seg_ids)



############## 3.BERTによる日本語ニュース分類 ##############
import glob  # ファイルの取得に使用
import sys
sys.getdefaultencoding()
import os
os.chdir('D:/GitHub/DS3')

# データセットの読み込み
path = "D:/GitHub/DS3/data/bert_nlp/text/"  # フォルダの場所を指定

dir_files = os.listdir(path=path)
dirs = [f for f in dir_files if os.path.isdir(os.path.join(path, f))]  # ディレクトリ一覧

text_label_data = []  # 文章とラベルのセット
dir_count = 0  # ディレクトリ数のカウント
file_count= 0  # ファイル数のカウント

for i in range(len(dirs)):
    dir = dirs[i]
    files = glob.glob(path + dir + "/*.txt")  # ファイルの一覧
    dir_count += 1

    for file in files:
        if os.path.basename(file) == "LICENSE.txt":
            continue

        with open(file, "r", encoding="utf-8") as f:
            text = f.readlines()[3:]
            text = "".join(text)
            text = text.translate(str.maketrans({"\n":"", "\t":"", "\r":"", "\u3000":""}))
            text_label_data.append([text, i])

        file_count += 1
        print("\rfiles: " + str(file_count) + "dirs: " + str(dir_count), end="")

# データの保存
import csv
from sklearn.model_selection import train_test_split

news_train, news_test =  train_test_split(text_label_data, shuffle=True)  # 訓練用とテスト用に分割
news_path = "D:/GitHub/DS3/data/bert_nlp/"

with open(news_path+"news_train.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(news_train)

with open(news_path+"news_test.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(news_test)

# モデルとTokenizerの読み込み
# https://github.com/cl-tohoku/bert-japanese

from transformers import BertForSequenceClassification, BertJapaneseTokenizer

sc_model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=10)
sc_model.cuda()
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

# データセットの読み込み
from datasets import load_dataset

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

# news_path = "/content/drive/My Drive/bert_nlp/section_5/"

train_data = load_dataset("csv", data_files=news_path+"news_train.csv", column_names=["text", "label"], split="train")
train_data = train_data.map(tokenize, batched=True, batch_size=len(train_data))
train_data.set_format("torch", columns=["input_ids", "label"])

test_data = load_dataset("csv", data_files=news_path+"news_test.csv", column_names=["text", "label"], split="train")
test_data = test_data.map(tokenize, batched=True, batch_size=len(test_data))
test_data.set_format("torch", columns=["input_ids", "label"])

# 評価用の関数
from sklearn.metrics import accuracy_score

def compute_metrics(result):
    labels = result.label_ids
    preds = result.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }

# Trainerの設定
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir = "./data/bert_nlp/results",
    num_train_epochs = 2,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 32,
    warmup_steps = 500,  # 学習係数が0からこのステップ数で上昇
    weight_decay = 0.01,  # 重みの減衰率
    logging_dir = "./data/bert_nlp/logs",
    evaluation_strategy = "steps"
)

trainer = Trainer(
    model = sc_model,
    args = training_args,
    compute_metrics = compute_metrics,
    train_dataset = train_data,
    eval_dataset = test_data,
)

# モデルの訓練
trainer.train()

# モデルの評価
trainer.evaluate()

# モデルの保存
sc_model.save_pretrained(news_path)
tokenizer.save_pretrained(news_path)
# モデルの読み込み
loaded_model = BertForSequenceClassification.from_pretrained(news_path)
loaded_model.cuda()
loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(news_path)


# 日本語ニュースの分類
import glob  # ファイルの取得に使用
import os
import torch

category = "movie-enter"
sample_path = "D:/GitHub/DS3/data/bert_nlp/text/"  # フォルダの場所を指定
files = glob.glob(sample_path + category + "/*.txt")  # ファイルの一覧
file = files[12]  # 適当なニュース

dir_files = os.listdir(path=sample_path)
dirs = [f for f in dir_files if os.path.isdir(os.path.join(sample_path, f))]  # ディレクトリ一覧

with open(file, "r", encoding="utf-8") as f:
    sample_text = f.readlines()[3:]
    sample_text = "".join(sample_text)
    sample_text = sample_text.translate(str.maketrans({"\n":"", "\t":"", "\r":"", "\u3000":""}))

print(sample_text)

max_length = 512
words = loaded_tokenizer.tokenize(sample_text)
word_ids = loaded_tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
word_tensor = torch.tensor([word_ids[:max_length]])  # テンソルに変換

x = word_tensor.cuda()  # GPU対応
y = loaded_model(x)  # 予測
pred = y[0].argmax(-1)  # 最大値のインデックス
print("result:", dirs[pred])

