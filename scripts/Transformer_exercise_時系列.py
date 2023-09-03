import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math



# 前処理
def forecast_seq(model, sequences):
  model.eval()
  forecast_seq = torch.Tensor(0)
  actual = torch.Tensor(0)

  with torch.no_grad():
    for i in range(0, len(sequences) - 1):
      data, target = get_batch(sequences, i, 1)
      output = model(data)
      forecast_seq = torch.cat((forecast_seq, output[-1].view(-1).cpu()), 0)
      actual = torch.cat((actual, target[-1].view(-1).cpu()), 0)

  return forecast_seq, actual

df = pd.read_csv("D:/GitHub/DS3/data/1668134144_55154200_6554.txt", encoding='utf8', sep='\t')

header_flg = False; #ヘッダ行出現フラグ
l = []
for r in df.iterrows():
  if r[0][0] != "日付":
    l.append(r[0])
  elif not header_flg: 
    l.append(r[0])
    header_flg = True

data = pd.DataFrame(l[1:], columns=l[0])

close = np.array([int(str(i).replace(',', '')) for i in data['終値'].tolist() if i is not np.nan])

close = close[::-1]

logreturn = np.diff(np.log(close)) # 対数差分
csum_logreturn = logreturn.cumsum() # 累積和

def get_org(x, csum_logreturn):
  return np.hstack((x[0], x[0]*np.exp(csum_logreturn)))

y = np.log(2)

fig, axs = plt.subplots(2, 1)
axs[0].plot(close, color='orange')
axs[0].set_title('終値')
axs[0].set_ylabel('終値')
axs[0].set_xlabel('日数')
axs[1].plot(csum_logreturn, color='green')
axs[1].set_title("終値の対数差分の累積和")
axs[1].set_xlabel('日数')
fig.tight_layout()
plt.show()

# Transformer
input_window = 30
output_window = 1
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.arange(0, 10, dtype=torch.float).unsqueeze(1).shape)


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=5000):
    super().__init__()
    self.dropout = nn.Dropout(p=0.1)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer("pe", pe)
  
  def forward(self, x):
    return self.dropout(x + self.pe[:x.size(0), :])


class TransformerModel(nn.Module):
  def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
    super().__init__()
    self.model_type = 'Transformer'
    self.src_mask = None
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.pos_encoder = PositionalEncoding(d_model=feature_size)
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    self.decoder = nn.Linear(feature_size, 1)
  
  def init_weights(self):
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform(-0.1, 0.1)

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz))==1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

  def forward(self, src):
    if self.src_mask is None or self.src_mask.size(0) != len(src):
      device = self.device
      mask = self._generate_square_subsequent_mask(len(src)).to(device)
      self.src_mask = mask
    src = self.pos_encoder(src)
    output = self.transformer_encoder(src, self.src_mask)
    output = self.decoder(output)
    return output


def create_inout_sequences(input_data, tw):
  inout_seq = []
  L = len(input_data)

  for i in range(L-tw):
    train_seq = input_data[i:i+tw]
    train_label = input_data[i+output_window: i + tw + output_window]
    inout_seq.append((train_seq, train_label))
  return torch.FloatTensor(inout_seq)


def get_data(data, split):
  series = data
  split = round(split*len(series))
  
  train_data = series[:split]
  train_data = train_data.cumsum()
  train_data = 2 * train_data

  test_data = series[split:]
  test_data = test_data.cumsum()

  train_sequence = create_inout_sequences(train_data, input_window)
  test_sequence = create_inout_sequences(test_data, input_window)
  
  return train_sequence.to(device), test_sequence.to(device)


torch.stack?


def get_batch(source, i, batch_size):
  """ (input_window, batch_size, output_window)
  """
  seq_len = min(batch_size, len(source)-1-i)
  data = source[i:i+seq_len]

  input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, output_window))
  target  = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, output_window))

  return input, target


def train(train_data):
  model.train()
  total_loss = 0.
  start_time = time.time()

  for batch, i in enumerate(range(0, len(train_data), batch_size)):
    data, targets = get_batch(train_data, i, batch_size)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    log_interval = int(len(train_data)/batch_size/5)

    if batch % log_interval == 0 and batch > 0:
      cur_loss = total_loss / log_interval
      elapsed = time.time() - start_time
      print(f'{epoch:3d}:epoch | {batch:5d}/{len(train_data)//batch_size:5d} batches | {elapsed*1000/log_interval:5.2f} ms | {cur_loss:5.7} : loss')

      total_loss = 0
      start_time = time.time()


def evaluate(eval_model, data_source):
  eval_model.eval()
  total_loss = 0.
  eval_batch_size = 16

  with torch.no_grad():
    for i in range(0, len(data_source), eval_batch_size):
      data, targets = get_batch(data_source, i, eval_batch_size)
      output = eval_model(data)
      total_loss += len(data[0]) * criterion(output, targets).cpu().item()

  return total_loss/len(data_source)


model = TransformerModel().to(device)

criterion = nn.MSELoss()
lr = 0.00005

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.95)

epochs = 100
batch_size = 16

train_data, val_data = get_data(logreturn, 0.6)

# 学習
for epoch in range(1, epochs + 1):
  epoch_start_time = time.time()
  train(train_data)

  if (epoch % epochs == 0):
    val_loss = evaluate(model, val_data)
    print(f'val loss:{val_loss:5.7f}')
  
  else:
    print(f'{epoch}:epoch | time: {time.time() - epoch_start_time:5.2f} sec')
  
  print('=====' * 80)

  scheduler.step()

def forecast_seq(model, sequences):
  model.eval()
  forecast_seq = torch.Tensor(0)
  actual = torch.Tensor(0)

  with torch.no_grad():
    for i in range(0, len(sequences) - 1):
      data, target = get_batch(sequences, i, 1)
      output = model(data)
      forecast_seq = torch.cat((forecast_seq, output[-1].view(-1).cpu()), 0)
      actual = torch.cat((actual, target[-1].view(-1).cpu()), 0)

  return forecast_seq, actual


val_data.shape

test_result, truth = forecast_seq(model, val_data)


plt.plot(truth, color='red', alpha=0.7)
plt.plot(test_result, color='blue', linewidth=0.7)
plt.title('Actual vs. Forecast')
plt.legend(['Actual', 'Forecast'])
plt.xlabel('Time step')
plt.show()




get_org([close[729]], test_result)

get_org([close[729]], truth)
