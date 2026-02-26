import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

data = pd.read_pickle(r"dl\data/jyrd深度学习选取参数_1min.pkl")

# 丢掉全空行
data = data.dropna(axis=0)

target_cols = [
    "总负荷",
    "GT1实发",
    "GT2实发",
    "汽机负荷"
]
feature_cols = list(data.columns[:-5])

data = data[(data[target_cols] > 0).all(axis=1)].copy()


# 转为数值（不可转的置 NaN）
X_df = data[feature_cols].apply(pd.to_numeric, errors="coerce")
y_df = data[target_cols].apply(pd.to_numeric, errors="coerce")

# 丢掉含 NaN 的行
df_all = pd.concat([X_df, y_df], axis=1).dropna()
X = df_all[feature_cols].to_numpy(dtype=np.float32)
y = df_all[target_cols].to_numpy(dtype=np.float32)

# scaler 分开
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# --------------------------
# 1) 划分：train / val / test（不打乱，按时间切）
#    先切出 test，再从 train 里切 val
# --------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, shuffle=False
)

# val 占 전체数据的 0.1（在 trainval 里再切 0.125 = 0.1 / 0.8）
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.125, shuffle=False
)

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

Epochs = 500
BatchSize = 64
LearningRate = 0.001

# 早停参数
patience = 5          # 连续多少轮 val 不提升就停
min_delta = 1e-6       # 认为“提升”的最小幅度

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True)

net = Net(in_dim=X_train.shape[1], out_dim=y_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_state = None
bad_count = 0

for epoch in range(Epochs):
    # ---- train ----
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

    # ---- val ----
    net.eval()
    with torch.no_grad():
        val_pred = net(X_val_tensor)
        val_loss = criterion(val_pred, y_val_tensor).item()
    val_losses.append(val_loss)

    print(f"Epoch {epoch:04d} TrainLoss: {epoch_train_loss:.6f} | ValLoss: {val_loss:.6f}")

    # ---- early stopping ----
    if best_val_loss - val_loss > min_delta:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        bad_count = 0
    else:
        bad_count += 1

    if bad_count >= patience:
        print(f"[EarlyStop] epoch={epoch}, best_val_loss={best_val_loss:.6f}")
        break

# 恢复最佳模型
if best_state is not None:
    net.load_state_dict(best_state)

# --------------------------
# 3) 测试集预测 + 指标
# --------------------------
net.eval()
with torch.no_grad():
    y_pred_test = net(X_test_tensor).cpu().numpy()

print("\n[Scaled Metrics]")
for i, name in enumerate(target_cols):
    mae = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
    mse = mean_squared_error(y_test[:, i], y_pred_test[:, i])
    r2  = r2_score(y_test[:, i], y_pred_test[:, i])
    print(f"{name}: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")

# 反归一化到原尺度
y_test_real = y_scaler.inverse_transform(y_test)
y_pred_real = y_scaler.inverse_transform(y_pred_test)

print("\n[Real-Scale Metrics]")
for i, name in enumerate(target_cols):
    mae = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_real[:, i], y_pred_real[:, i]))
    r2  = r2_score(y_test_real[:, i], y_pred_real[:, i])
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# --------------------------
# 4) 画图：实际值 vs 预测值（每个输出一张）
#    用原尺度画，更直观
# --------------------------

fig, axes = plt.subplots(len(target_cols), 1, sharex=True, figsize=(12, 8))

for i, name in enumerate(target_cols):
    axes[i].plot(y_test_real[:, i], label="Actual")
    axes[i].plot(y_pred_real[:, i], label="Pred")
    axes[i].set_ylabel(name)
    axes[i].legend()
    axes[i].grid(True)

axes[-1].set_xlabel("Time index (test)")
plt.show()
