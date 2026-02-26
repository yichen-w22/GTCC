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
# 0) 读数据（CSV）
# =========================
data_path = r"D:\清华\毕业设计\test\dl\data\PINN选取参数_1min.csv"
data = pd.read_csv(data_path)

target_cols = ["总负荷", "GT1实发", "GT2实发", "汽机负荷"]


# -------------------------
# 1) 单位转换（你原样保留）
# -------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

data["大气气温"] = to_num(data["大气气温"])
data["大气气压"] = to_num(data["大气气压"])
T_amb_K = data["大气气温"] + 273.15
P_amb_Pa = data["大气气压"] * 1000.0

def F_to_K(x):
    return (x - 32.0) * 5.0/9.0 + 273.15

T_ci1_K   = F_to_K(to_num(data["Compressor Inlet Temperature"]))
T_exh1_K  = F_to_K(to_num(data["Exhaust Temp MeDXan Corrected By Average"]))
T_fg1_K   = F_to_K(to_num(data["Fuel gas temperature"]))

T_ci2_K   = F_to_K(to_num(data["Compressor Inlet Temperature.1"]))
T_exh2_K  = F_to_K(to_num(data["Exhaust Temp MeDXan Corrected By Average.1"]))
T_fg2_K   = F_to_K(to_num(data["Fuel gas temperature.1"]))

T_stack1_K = (to_num(data["#1锅炉出口排烟温度1"]) +
              to_num(data["#1锅炉出口排烟温度2"]) +
              to_num(data["#1锅炉出口排烟温度3"])) / 3.0 + 273.15

T_stack2_K = (to_num(data["#2锅炉出口排烟温度2"]) +
              to_num(data["#2锅炉出口排烟温度3"])) / 2.0 + 273.15

def t_per_h_to_kg_per_s(x):
    return x * 1000.0 / 3600.0

m_f1 = t_per_h_to_kg_per_s(to_num(data["Gas Fuel Flow"]))
m_g1 = t_per_h_to_kg_per_s(to_num(data["Turbine Exhaust Mass Flow"]))
m_f2 = t_per_h_to_kg_per_s(to_num(data["Gas Fuel Flow.1"]))
m_g2 = t_per_h_to_kg_per_s(to_num(data["Turbine Exhaust Mass Flow.1"]))

p_exh_st_abs_Pa = (to_num(data["低压排汽压力(外缸调端)#1"]) + data["大气气压"]) * 1000.0


# -------------------------
# 2) 仅对目标列做过滤（你原样）
# -------------------------
y_df = data[target_cols].apply(to_num)
mask_y = (y_df.notna().all(axis=1)) & (y_df > 0).all(axis=1)

data = data.loc[mask_y].reset_index(drop=True)
y_df = y_df.loc[mask_y].reset_index(drop=True)

T_amb_K = T_amb_K.loc[mask_y].reset_index(drop=True)
T_ci1_K = T_ci1_K.loc[mask_y].reset_index(drop=True)
T_exh1_K = T_exh1_K.loc[mask_y].reset_index(drop=True)
T_ci2_K = T_ci2_K.loc[mask_y].reset_index(drop=True)
T_exh2_K = T_exh2_K.loc[mask_y].reset_index(drop=True)
T_stack1_K = T_stack1_K.loc[mask_y].reset_index(drop=True)
T_stack2_K = T_stack2_K.loc[mask_y].reset_index(drop=True)
m_f1 = m_f1.loc[mask_y].reset_index(drop=True)
m_g1 = m_g1.loc[mask_y].reset_index(drop=True)
m_f2 = m_f2.loc[mask_y].reset_index(drop=True)
m_g2 = m_g2.loc[mask_y].reset_index(drop=True)


# -------------------------
# 3) 特征：数值化 + 删全NaN列 + 中位数填充（你之前已做）
# -------------------------
feature_cols = [c for c in data.columns if c not in target_cols]
X_df = data[feature_cols].apply(to_num)

X_df = X_df.dropna(axis=1, how="all")
X_df = X_df.apply(lambda col: col.fillna(col.median()), axis=0)
X_df = X_df.fillna(0.0)

X = X_df.to_numpy(dtype=np.float32)
y = y_df.to_numpy(dtype=np.float32)

# -------------------------
# 3.5 关键修复：phys 同样做“有限值填充”，否则 loss 会 NaN
# -------------------------
phys_df = pd.DataFrame({
    "Tref":  T_amb_K,
    "Tci1":  T_ci1_K,
    "Texh1": T_exh1_K,
    "Tst1":  T_stack1_K,
    "mf1":   m_f1,
    "mg1":   m_g1,
    "Tci2":  T_ci2_K,
    "Texh2": T_exh2_K,
    "Tst2":  T_stack2_K,
    "mf2":   m_f2,
    "mg2":   m_g2,
}).apply(to_num)

# Inf -> NaN
phys_df = phys_df.replace([np.inf, -np.inf], np.nan)
# 中位数填充
phys_df = phys_df.apply(lambda col: col.fillna(col.median()), axis=0).fillna(0.0)

phys = phys_df.to_numpy(dtype=np.float32)

print("[CHECK] X shape:", X.shape, "y shape:", y.shape, "phys shape:", phys.shape)
print("[CHECK] phys finite:", np.isfinite(phys).all())


# =========================
# 4) scaler 分开（结构不变）
# =========================
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# 时间切分：train/val/test
X_trainval, X_test, y_trainval, y_test, phys_trainval, phys_test = train_test_split(
    X_scaled, y_scaled, phys, test_size=0.2, shuffle=False
)
X_train, X_val, y_train, y_val, phys_train, phys_val = train_test_split(
    X_trainval, y_trainval, phys_trainval, test_size=0.125, shuffle=False
)


# =========================
# 5) 网络（结构不变）
# =========================
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, out_dim)

        self.b_gt1 = nn.Parameter(torch.zeros(1))
        self.b_gt2 = nn.Parameter(torch.zeros(1))
        self.b_st  = nn.Parameter(torch.zeros(1))
        self.k_cond_raw = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return self.fc4(h)

    def k_cond(self):
        return torch.nn.functional.softplus(self.k_cond_raw)


# =========================
# 6) PINN loss（加：nan_to_num + clamp 防爆）
# =========================
# 更稳的反归一化：y_real = y_scaled / scale + data_min
y_scale_t = torch.tensor(y_scaler.scale_, dtype=torch.float32)
y_min_data_t = torch.tensor(y_scaler.data_min_, dtype=torch.float32)

def inv_y(y_scaled_t):
    return y_scaled_t / y_scale_t + y_min_data_t

cp_air = 1.005  # kJ/kg-K
cp_gas = 1.150  # kJ/kg-K
LHV = 47.5      # MJ/kg

def mw_from_mcpdt(m, cp_kj, dT):
    return (m * cp_kj * dT) / 1000.0  # kW->MW

def loss_pinn(net, y_pred_scaled, y_true_scaled, phys_batch, lam_gt=0.2, lam_st=0.2):
    # 监督项（scaled）
    L_data = torch.mean((y_pred_scaled - y_true_scaled) ** 2)

    # 反归一化（MW）
    y_pred = inv_y(y_pred_scaled)
    P_gt1 = y_pred[:, 1]
    P_gt2 = y_pred[:, 2]
    P_st  = y_pred[:, 3]

    # phys: [Tref,Tci1,Texh1,Tst1,mf1,mg1,Tci2,Texh2,Tst2,mf2,mg2]
    Tref  = phys_batch[:, 0]
    Tci1  = phys_batch[:, 1]
    Texh1 = phys_batch[:, 2]
    Tst1  = phys_batch[:, 3]
    mf1   = phys_batch[:, 4]
    mg1   = phys_batch[:, 5]

    Tci2  = phys_batch[:, 6]
    Texh2 = phys_batch[:, 7]
    Tst2  = phys_batch[:, 8]
    mf2   = phys_batch[:, 9]
    mg2   = phys_batch[:,10]

    # -------- GT 能量守恒（MW）--------
    ma1 = torch.clamp(mg1 - mf1, min=0.0)
    ma2 = torch.clamp(mg2 - mf2, min=0.0)

    Ein_air1  = mw_from_mcpdt(ma1, cp_air, (Tci1 - Tref))
    Ein_air2  = mw_from_mcpdt(ma2, cp_air, (Tci2 - Tref))
    Ein_fuel1 = mf1 * LHV
    Ein_fuel2 = mf2 * LHV
    Eout_exh1 = mw_from_mcpdt(mg1, cp_gas, (Texh1 - Tref))
    Eout_exh2 = mw_from_mcpdt(mg2, cp_gas, (Texh2 - Tref))

    r_gt1 = (Ein_air1 + Ein_fuel1) - (Eout_exh1 + P_gt1 + net.b_gt1)
    r_gt2 = (Ein_air2 + Ein_fuel2) - (Eout_exh2 + P_gt2 + net.b_gt2)

    # 防爆：把残差裁剪到合理范围（MW）
    r_gt1 = torch.clamp(r_gt1, -5e4, 5e4)
    r_gt2 = torch.clamp(r_gt2, -5e4, 5e4)

    # -------- ST 能量守恒（MW）--------
    Qdrop = mw_from_mcpdt(mg1, cp_gas, (Texh1 - Tst1)) + mw_from_mcpdt(mg2, cp_gas, (Texh2 - Tst2))
    kcond = net.k_cond()
    r_st = Qdrop - ((1.0 + kcond) * P_st + net.b_st)
    r_st = torch.clamp(r_st, -5e4, 5e4)

    # nan/inf 兜底
    r_gt1 = torch.nan_to_num(r_gt1, nan=0.0, posinf=0.0, neginf=0.0)
    r_gt2 = torch.nan_to_num(r_gt2, nan=0.0, posinf=0.0, neginf=0.0)
    r_st  = torch.nan_to_num(r_st,  nan=0.0, posinf=0.0, neginf=0.0)

    L_gt = torch.mean((r_gt1/1000.0)**2 + (r_gt2/1000.0)**2)
    L_st = torch.mean((r_st/1000.0)**2)

    return L_data + lam_gt * L_gt + lam_st * L_st


# =========================
# 7) 训练（加：梯度裁剪，防止爆炸）
# =========================
Epochs = 500
BatchSize = 64
LearningRate = 0.001
patience = 5
min_delta = 1e-6

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
phys_train_tensor = torch.tensor(phys_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
phys_val_tensor = torch.tensor(phys_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor, phys_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)

net = Net(in_dim=X_train.shape[1], out_dim=y_train.shape[1])
optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)

best_val_loss = float("inf")
best_state = None
bad_count = 0

for epoch in range(Epochs):
    net.train()
    epoch_train_loss = 0.0

    for X_batch, y_batch, phys_batch in train_loader:
        optimizer.zero_grad()
        y_pred = net(X_batch)
        loss = loss_pinn(net, y_pred, y_batch, phys_batch, lam_gt=0.2, lam_st=0.2)
        loss.backward()

        # 关键：梯度裁剪，避免一旦爆炸就全NaN
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

        optimizer.step()
        epoch_train_loss += loss.item() * len(X_batch)

    epoch_train_loss /= len(train_dataset)

    net.eval()
    with torch.no_grad():
        yv_pred = net(X_val_tensor)
        val_loss = loss_pinn(net, yv_pred, y_val_tensor, phys_val_tensor, lam_gt=0.2, lam_st=0.2).item()

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
# 8) 测试集评估 + 画图
# =========================
net.eval()
with torch.no_grad():
    y_pred_test = net(X_test_tensor).cpu().numpy()

# 若仍有NaN，先兜底处理（避免 sklearn 报错）
y_pred_test = np.nan_to_num(y_pred_test, nan=0.0, posinf=0.0, neginf=0.0)

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

fig, axes = plt.subplots(len(target_cols), 1, sharex=True, figsize=(12, 8))
for i, name in enumerate(target_cols):
    axes[i].plot(y_test_real[:, i], label="Actual")
    axes[i].plot(y_pred_real[:, i], label="Pred")
    axes[i].set_ylabel(name)
    axes[i].legend()
    axes[i].grid(True)
axes[-1].set_xlabel("Time index (test)")
plt.show()

print("\n[Learned PINN params]")
print("b_gt1(MW)=", float(net.b_gt1.detach().cpu().numpy()))
print("b_gt2(MW)=", float(net.b_gt2.detach().cpu().numpy()))
print("b_st(MW) =", float(net.b_st.detach().cpu().numpy()))
print("k_cond   =", float(net.k_cond().detach().cpu().numpy()))
