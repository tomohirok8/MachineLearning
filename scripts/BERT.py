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



