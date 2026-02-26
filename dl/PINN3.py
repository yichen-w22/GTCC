import pandas as pd

# Load data (assuming CSV format for demonstration)
df = pd.read_csv("D:\清华\毕业设计\test\dl\data\PINN选取参数_1min.csv")
# Identify relevant columns by keywords (for brevity, listing them directly):
control_cols = ["Gas Fuel Flow", "Gas Fuel Flow.1", "IGV angle in deg", "IGV angle in deg.1"]  # etc.
state_cols = [col for col in df.columns if col not in control_cols + ["总负荷","GT1实发","GT2实发","汽机负荷"]]
target_cols = ["总负荷", "GT1实发", "GT2实发", "汽机负荷"]

# Convert units:
# Pressures to Pa (add atmosphere for gauge)
atm_pressure_Pa = df["大气气压"] * 1000.0  # assuming ambient was in kPa
for col in df.columns:
    if "压力" in col and col != "大气气压":
        # assume MPa if values ~0-10, kPa if ~0-100, or gauge negative values
        if df[col].abs().max() < 500:  # values like 5.6 (MPa) or 100 (kPa) fall here
            if df[col].mean() < 50:  # likely MPa (around 5) vs kPa (~100)
                df[col] = df[col]*1e6 + atm_pressure_Pa  # MPa to Pa, add atm for gauge
            else:
                df[col] = df[col]*1000 + atm_pressure_Pa  # kPa to Pa (if any gauge offset)
        else:
            df[col] = df[col] + atm_pressure_Pa  # already in Pa? (unlikely here)
# Temperatures to K
for col in df.columns:
    if col.endswith(("气温","温度", "Temperature")):  # ambient temp, gas temp, etc.
        df[col] = df[col] + 273.15
# Flow rates to kg/s
flow_cols = [c for c in df.columns if c.endswith("流量") or "Mass Flow" in c or "Fuel Flow" in c]
for col in flow_cols:
    df[col] = df[col] / 3.6  # convert t/h to kg/s (if already kg/s, this would need adjustment)

# Remove unphysical data:
# e.g. drop rows where any target or fuel flow is negative
df = df[(df[target_cols] >= 0).all(axis=1)]

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Define the MLP architecture (from provided Net class)
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
        return self.fc4(x)

# Convert training data to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# Initialize network and optimizer
model = Net(in_dim=X_train.shape[1], out_dim=y_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Constants for physics (using units: fuel in MJ/kg, cp in MJ/(kg·K), so outputs in MW)
LHV = 47.5        # MJ per kg of fuel
cp_gas = 1.15e-3  # ~1.15 kJ/kg·K = 0.00115 MJ/kg·K
cond_dh = 2.26    # MJ per kg steam (condensing enthalpy)

# Indices of relevant features in X (determined from columns order)
idx_fuel1 = list(df.columns).index("Gas Fuel Flow")        # fuel flow GT1 (kg/s)
idx_fuel2 = list(df.columns).index("Gas Fuel Flow.1")      # fuel flow GT2
idx_mexh1 = list(df.columns).index("Turbine Exhaust Mass Flow")    # GT1 exhaust mass flow (kg/s)
idx_mexh2 = list(df.columns).index("Turbine Exhaust Mass Flow.1")  # GT2 exhaust mass flow
idx_texh1_in = list(df.columns).index("Exhaust Temp MeDXan Corrected By Average")   # GT1 exhaust inlet temp (K)
idx_texh1_out = list(df.columns).index("#1锅炉出口排烟温度3")      # GT1 stack exit temp (K)
idx_texh2_in = list(df.columns).index("Exhaust Temp MeDXan Corrected By Average.1") # GT2 exhaust inlet temp (K)
idx_texh2_out = list(df.columns).index("#2锅炉出口排烟温度3")      # GT2 stack exit temp (K)
idx_feed = list(df.columns).index("凝结水泵出口母管凝结水流量")      # feedwater flow (kg/s)

# Training loop
lambda1, lambda2 = 0.1, 0.1  # physics loss weights (can be tuned)
criterion = nn.MSELoss()
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        # Supervised data loss (MSE on the 4 outputs)
        data_loss = criterion(y_pred, y_batch)
        # Compute physics residuals for each sample in the batch:
        fuel1 = X_batch[:, idx_fuel1]      # (batch,)
        fuel2 = X_batch[:, idx_fuel2]
        m_exh1 = X_batch[:, idx_mexh1]
        m_exh2 = X_batch[:, idx_mexh2]
        T_exh1_in = X_batch[:, idx_texh1_in]
        T_exh1_out = X_batch[:, idx_texh1_out]
        T_exh2_in = X_batch[:, idx_texh2_in]
        T_exh2_out = X_batch[:, idx_texh2_out]
        feed_flow = X_batch[:, idx_feed]
        # Predicted outputs:
        P_total = y_pred[:, 0]  # (not used in physics directly)
        P_gt1   = y_pred[:, 1]
        P_gt2   = y_pred[:, 2]
        P_st    = y_pred[:, 3]
        # Gas turbine energy residuals (in MW):
        res_gt1 = fuel1 * LHV - (P_gt1 + m_exh1 * cp_gas * (T_exh1_in - T_exh1_out))
        res_gt2 = fuel2 * LHV - (P_gt2 + m_exh2 * cp_gas * (T_exh2_in - T_exh2_out))
        # Steam cycle energy residual (in MW):
        Q_hrsg_total = m_exh1 * cp_gas * (T_exh1_in - T_exh1_out) + m_exh2 * cp_gas * (T_exh2_in - T_exh2_out)
        res_st = Q_hrsg_total - (P_st + feed_flow * cond_dh)
        # Physics loss (MSE of residuals):
        GT_energy_res = torch.mean(res_gt1**2 + res_gt2**2)   # combined GT1+GT2 residual
        ST_energy_res = torch.mean(res_st**2)
        phys_loss = lambda1 * GT_energy_res + lambda2 * ST_energy_res
        # Total loss and backpropagation
        loss = data_loss + phys_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # (Optional) print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: total loss = {running_loss/len(train_loader):.4f}, data_loss = {data_loss.item():.4f}, phys_loss = {phys_loss.item():.4f}")
