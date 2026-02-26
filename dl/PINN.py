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
# 0) 读数据
# =========================
data = pd.read_pickle(r"dl\data/jyrd深度学习选取参数_1min.pkl")
cols = pd.Series(data.columns)
data.columns = cols + cols.groupby(cols).cumcount().replace(0, "").astype(str).radd(".").replace(".", "")

# 丢掉全空行
data = data.dropna(axis=0)

target_cols = ["总负荷", "GT1实发", "GT2实发", "汽机负荷"]
feature_cols = list(data.columns[:-5])

# 删除目标值中含非正值的行
data = data[(data[target_cols] > 0).all(axis=1)].copy()

# 转为数值（不可转置 NaN）
X_df = data[feature_cols].apply(pd.to_numeric, errors="coerce")
y_df = data[target_cols].apply(pd.to_numeric, errors="coerce")

# 丢掉含 NaN 的行（确保对齐）
df_all = pd.concat([X_df, y_df], axis=1).dropna()

X = df_all[feature_cols].to_numpy(dtype=np.float32)
y = df_all[target_cols].to_numpy(dtype=np.float32)


# =========================
# 1) 取燃料流量
# =========================
FUEL_COL_GT1 = "Gas Fuel Flow.1"
FUEL_COL_GT2 = "Gas Fuel Flow"

fuel1 = df_all[FUEL_COL_GT1].to_numpy(dtype=np.float32)
fuel2 = df_all[FUEL_COL_GT2].to_numpy(dtype=np.float32)
F = np.stack([fuel1, fuel2], axis=1).astype(np.float32)  # (N,2)


# =========================
# 2) scaler 分开
# =========================
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)


# =========================
# 3) 划分：train / val / test（不打乱，按时间切）
# =========================
X_trainval, X_test, y_trainval, y_test, F_trainval, F_test = train_test_split(
    X_scaled, y_scaled, F, test_size=0.2, shuffle=False
)

X_train, X_val, y_train, y_val, F_train, F_val = train_test_split(
    X_trainval, y_trainval, F_trainval, test_size=0.125, shuffle=False
)


# =========================
# 4) PINN 网络：输出功率 + 两台燃机等效效率 eta
# =========================
class NetPINN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NetPINN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)

        # 功率头：4个输出（与原来一样）
        self.fc_out = nn.Linear(32, out_dim)

        # 效率头：2个输出（GT1, GT2）
        self.fc_eta = nn.Linear(32, 2)
        self.eta_min = 0.15
        self.eta_max = 0.45

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))

        y_scaled = self.fc_out(h)

        # eta 限制在合理范围（避免学出离谱效率）
        eta01 = torch.sigmoid(self.fc_eta(h))
        eta = self.eta_min + (self.eta_max - self.eta_min) * eta01
        return y_scaled, eta


# =========================
# 5) PINN 损失函数（尽量简单）
#    L = 数据拟合 + 燃料热输入约束 + 总功率平衡 + 非负
# =========================
# 将 y_scaler 参数转 torch，方便在 loss 内“可微反归一化”
y_scale_t = torch.tensor(y_scaler.scale_, dtype=torch.float32)  # shape(4,)
y_min_t   = torch.tensor(y_scaler.min_,   dtype=torch.float32)

def inv_y(y_scaled_t):
    # sklearn: y_real = (y_scaled - min_) / scale_
    return (y_scaled_t - y_min_t) / y_scale_t


# ====== 关键：热输入换算（你需要根据单位改一行）======
# 默认：Gas Fuel Flow 单位 = kg/s，LHV = 50 MJ/kg，则 Qin(MW)=m_dot*LHV
LHV_MJ_per_kg = 50.0

# 若你的 Gas Fuel Flow 单位是 Nm3/h，可改成：
# LHV_MJ_per_Nm3 = 35.8
# Qin = Vdot(Nm3/h) * LHV(MJ/Nm3) / 3600

def loss_pinn(y_pred_scaled, y_true_scaled, eta, F_batch,
              lambda_gt=1.0, lambda_bal=0.2, lambda_nonneg=0.05):
    # 1) 数据拟合（scaled 空间）
    L_data = torch.mean((y_pred_scaled - y_true_scaled) ** 2)

    # 2) 反归一化到原尺度（MW），做物理约束
    y_pred_real = inv_y(y_pred_scaled)

    P_total = y_pred_real[:, 0]
    P_gt1   = y_pred_real[:, 1]
    P_gt2   = y_pred_real[:, 2]
    P_st    = y_pred_real[:, 3]

    # 3) 热输入 Qin（MW）
    fuel1 = F_batch[:, 0]
    fuel2 = F_batch[:, 1]

    Qin1 = fuel1 * LHV_MJ_per_kg
    Qin2 = fuel2 * LHV_MJ_per_kg

    eta1 = eta[:, 0]
    eta2 = eta[:, 1]

    # 4) 燃机物理残差：P_gt ≈ eta * Qin
    r_gt1 = P_gt1 - eta1 * Qin1
    r_gt2 = P_gt2 - eta2 * Qin2
    L_gt = torch.mean(r_gt1**2 + r_gt2**2)

    # 5) 总功率平衡：总负荷 ≈ GT1 + GT2 + 汽机
    r_bal = P_total - (P_gt1 + P_gt2 + P_st)
    L_bal = torch.mean(r_bal**2)

    # 6) 非负约束：P >= 0
    P_stack = torch.stack([P_total, P_gt1, P_gt2, P_st], dim=1)
    L_nonneg = torch.mean(torch.relu(-P_stack) ** 2)

    L = L_data + lambda_gt * L_gt + lambda_bal * L_bal + lambda_nonneg * L_nonneg
    return L


# =========================
# 6) 训练配置
# =========================
Epochs = 500
BatchSize = 64
LearningRate = 0.001

# 早停参数
patience = 5
min_delta = 1e-6

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
F_train_tensor = torch.tensor(F_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
F_val_tensor = torch.tensor(F_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
F_test_tensor = torch.tensor(F_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor, F_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True)

net = NetPINN(in_dim=X_train.shape[1], out_dim=y_train.shape[1])
optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)

best_val_loss = float("inf")
best_state = None
bad_count = 0

for epoch in range(Epochs):
    # ---- train ----
    net.train()
    epoch_train_loss = 0.0
    for X_batch, y_batch, F_batch in train_loader:
        optimizer.zero_grad()
        y_pred_scaled, eta = net(X_batch)
        loss = loss_pinn(y_pred_scaled, y_batch, eta, F_batch,
                         lambda_gt=1.0, lambda_bal=0.2, lambda_nonneg=0.05)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * len(X_batch)

    epoch_train_loss /= len(train_dataset)

    # ---- val ----
    net.eval()
    with torch.no_grad():
        yv_pred, eta_v = net(X_val_tensor)
        val_loss = loss_pinn(yv_pred, y_val_tensor, eta_v, F_val_tensor,
                             lambda_gt=1.0, lambda_bal=0.2, lambda_nonneg=0.05).item()

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


# =========================
# 7) 测试集预测 + 指标（保持你原来的写法）
# =========================
net.eval()
with torch.no_grad():
    y_pred_test_scaled, eta_test = net(X_test_tensor)
    y_pred_test = y_pred_test_scaled.cpu().numpy()

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


# =========================
# 8) 画图：实际值 vs 预测值（保持你的结构）
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


# =========================
# 9)（可选）看一下学到的效率范围（可解释性）
# =========================
eta_np = eta_test.cpu().numpy()
print("\n[Learned eta range on test]")
print(f"GT1 eta: min={eta_np[:,0].min():.3f}, mean={eta_np[:,0].mean():.3f}, max={eta_np[:,0].max():.3f}")
print(f"GT2 eta: min={eta_np[:,1].min():.3f}, mean={eta_np[:,1].mean():.3f}, max={eta_np[:,1].max():.3f}")
