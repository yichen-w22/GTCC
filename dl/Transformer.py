import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader


# =========================
# 0) 读数据 & 选列
# =========================
data = pd.read_pickle(r"dl\data/jyrd深度学习选取参数_1min.pkl")
data = data.dropna(axis=0)

target_cols = ["总负荷", "GT1实发", "GT2实发", "汽机负荷"]
feature_cols = list(data.columns[:-5])  # 你也可以改成自己手动选列名 list

data = data[(data[target_cols] > 0).all(axis=1)].copy()

X_df = data[feature_cols].apply(pd.to_numeric, errors="coerce")
y_df = data[target_cols].apply(pd.to_numeric, errors="coerce")

df_all = pd.concat([X_df, y_df], axis=1).dropna()
X = df_all[feature_cols].to_numpy(dtype=np.float32)
y = df_all[target_cols].to_numpy(dtype=np.float32)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)


# =========================
# 1) 生成序列：X_seq -> y
# =========================
seq_len = 30  # 用过去30分钟预测当前（可改 60/120）

def make_sequences(X_arr, y_arr, seq_len):
    Xs, ys = [], []
    for t in range(seq_len - 1, len(X_arr)):
        Xs.append(X_arr[t - seq_len + 1: t + 1, :])  # (seq_len, n_features)
        ys.append(y_arr[t, :])                        # (n_targets,)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_seq, y_seq = make_sequences(X_scaled, y_scaled, seq_len)
print("X_seq shape:", X_seq.shape, "y_seq shape:", y_seq.shape)


# =========================
# 2) 划分 train / val / test（不打乱）
# =========================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.125, shuffle=False
)


# =========================
# 3) Transformer 模型
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        x = self.input_proj(x)    # -> (batch, seq_len, d_model)
        x = self.pos_enc(x)       # 加位置编码
        h = self.encoder(x)       # -> (batch, seq_len, d_model)
        last = h[:, -1, :]        # 取最后一步
        yhat = self.head(last)    # -> (batch, out_dim)
        return yhat


# =========================
# 4) 训练设置（保持你原风格）
# =========================
Epochs = 500
BatchSize = 64
LearningRate = 0.001

# Transformer 超参（先用保守小模型，别一上来太大）
d_model = 64
nhead = 4
num_layers = 2
dim_ff = 128
dropout = 0.1

patience = 5
min_delta = 1e-6

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True)

net = TransformerRegressor(
    in_dim=X_train.shape[2],
    out_dim=y_train.shape[1],
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_ff=dim_ff,
    dropout=dropout
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)

train_losses, val_losses = [], []
best_val_loss = float("inf")
best_state = None
bad_count = 0


# =========================
# 5) 训练 + 早停
# =========================
for epoch in range(Epochs):
    net.train()
    epoch_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = net(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * len(X_batch)

    epoch_train_loss /= len(train_dataset)
    train_losses.append(epoch_train_loss)

    net.eval()
    with torch.no_grad():
        val_pred = net(X_val_tensor)
        val_loss = criterion(val_pred, y_val_tensor).item()
    val_losses.append(val_loss)

    print(f"Epoch {epoch:04d} TrainLoss: {epoch_train_loss:.6f} | ValLoss: {val_loss:.6f}")

    if best_val_loss - val_loss > min_delta:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        bad_count = 0
    else:
        bad_count += 1

    if bad_count >= patience:
        print(f"[EarlyStop] epoch={epoch}, best_val_loss={best_val_loss:.6f}")
        break

if best_state is not None:
    net.load_state_dict(best_state)


# =========================
# 6) 测试集预测 + 指标
# =========================
net.eval()
with torch.no_grad():
    y_pred_test = net(X_test_tensor).cpu().numpy()

print("\n[Scaled Metrics]")
for i, name in enumerate(target_cols):
    mae = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
    mse = mean_squared_error(y_test[:, i], y_pred_test[:, i])
    r2  = r2_score(y_test[:, i], y_pred_test[:, i])
    print(f"{name}: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")

y_test_real = y_scaler.inverse_transform(y_test)
y_pred_real = y_scaler.inverse_transform(y_pred_test)

print("\n[Real-Scale Metrics]")
for i, name in enumerate(target_cols):
    mae = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_real[:, i], y_pred_real[:, i]))
    r2  = r2_score(y_test_real[:, i], y_pred_real[:, i])
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")


# =========================
# 7) 画图：同一张图（子图）实际 vs 预测（原尺度）
# =========================
fig, axes = plt.subplots(len(target_cols), 1, sharex=True, figsize=(12, 8))

for i, name in enumerate(target_cols):
    axes[i].plot(y_test_real[:, i], label="Actual")
    axes[i].plot(y_pred_real[:, i], label="Pred")
    axes[i].set_ylabel(name)
    axes[i].legend()
    axes[i].grid(True)

axes[-1].set_xlabel("Time index (test)")
plt.show()
