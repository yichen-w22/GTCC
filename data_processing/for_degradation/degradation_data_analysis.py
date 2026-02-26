import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
p = BASE_DIR / Path(r"datareader_new\jqrd\outcome\jqrd燃机1_1min.pkl")

# ===== 降采样参数（只改这里）=====
# 每隔多少个缺失事件取 1 个（越大越稀疏）
EVENT_STRIDE = 50

# 如果你只想看部分特征，把列表填上；None 表示全看
FEATURE_WHITELIST = None
# 例如：
# FEATURE_WHITELIST = ["comp_inlet_T_K", "comp_inlet_P_MPa", "comp_exit_T_K", "comp_exit_P_MPa", "eta_isentropic"]
# =================================


# ===== 1) 读数据 & 处理重复列名 =====
df_raw = pd.read_pickle(p)
cols = pd.Series(df_raw.columns)
df_raw.columns = cols + cols.groupby(cols).cumcount().replace(0, "").astype(str).radd(".").replace(".", "")

df = pd.DataFrame(index=df_raw.index)
df["timestamp"] = df_raw.index

df["TEMP AMBIENT AIR"] = df_raw["TEMP AMBIENT AIR"]
df["REL HUMDITY AMB AIR"] = df_raw["REL HUMDITY AMB AIR"]

cols_comp_inlet_T_K = [
    "GT COMPR INLET AIR TEMP",
    "TEMP COMPR INLET GT1",
    "TEMP COMPR INLET",
    "TEMP COMPR INLET.1",
    "T COMPR INLET AVE",
    "TEMP COMPR INLET.2",
    "TEMP COMPR INLET.3",
    "TEMP COMPR INLET.4"
]
df["comp_inlet_T_K"] = df_raw[cols_comp_inlet_T_K].mean(axis=1) + 273.15
df["comp_inlet_P_MPa"] = df_raw["PRES U/STR COMPR"] * 0.0001

cols_comp_exit_T_K = ["TEMP COMPR OUTLET", "TEMP COMPR OUTLET.1", "TEMP COMPR OUTLET.2"]
df["comp_exit_T_K"] = df_raw[cols_comp_exit_T_K].mean(axis=1) + 273.15

cols_comp_exit_P_MPa = ["PRES COMPR OUTLET", "PRES COMPR OUTLET.1", "PRES COMPR OUTLET.2"]
df["comp_exit_P_MPa"] = df_raw[cols_comp_exit_P_MPa].mean(axis=1)

df["IGV_deg"] = df_raw["ACTUAL POSN IGV"]

cols_rps = ["TURBINE SPEED", "TURBINE SPEED.1"]
col_rpm = "TURBINE SPEED.2"
df["shaft_rpm"] = (df_raw[cols_rps].sum(axis=1) * 60.0 + df_raw[col_rpm]) / 3

cols1 = df_raw.columns[df_raw.columns.str.contains("TEMP TURB OUTLET", na=False)]
cols2 = df_raw.columns[df_raw.columns.str.contains("TEMP EXHAUST DUCT", na=False)]
cols_turbine_exhaust_T_K = cols1.tolist() + cols2.tolist()
df["turbine_exhaust_T_K"] = df_raw[cols_turbine_exhaust_T_K].mean(axis=1) + 273.15

cols_turbine_exit_P = ["进口烟压", "进口烟压2", "进口烟压3", "进口烟压4"]
df["turbine_exit_P_kPa"] = df_raw[cols_turbine_exit_P].mean(axis=1)

df["unit_power_MW"] = df_raw["APPARENT POWER"]
df["turbine_exhaust_flow"] = df_raw["余热锅炉出口烟气流量"]

# ===== 2) 记录 raw 缺失（此时还没有 eta_isentropic）=====
raw_nan_mask = df.isna().copy()

# ===== 3) 计算等熵效率 =====
gamma = 1.385
T1 = df["comp_inlet_T_K"]
T2 = df["comp_exit_T_K"]
P1 = df["comp_inlet_P_MPa"]
P2 = df["comp_exit_P_MPa"]

PR = P2 / P1
T2s = T1 * (PR ** ((gamma - 1.0) / gamma))
df["eta_isentropic"] = (T2s - T1) / (T2 - T1)

# ===== 4) 插值（只对部分列）=====
cols_interp = [
    "TEMP AMBIENT AIR",
    "REL HUMDITY AMB AIR",
    "comp_inlet_T_K",
    "comp_inlet_P_MPa",
    "IGV_deg"
]
df[cols_interp] = df[cols_interp].apply(
    lambda s: s.interpolate(method="linear", limit=9, limit_area="inside")
)

# ===== 5) 记录处理后缺失状态 -> derived =====
pre_drop_nan_mask = df.isna().copy()
raw_nan_mask = raw_nan_mask.reindex(columns=df.columns, fill_value=False)
derived_nan_mask = pre_drop_nan_mask & (~raw_nan_mask)

# ===== 6) 生成缺失事件表 =====
ts = pd.to_datetime(df["timestamp"], errors="coerce")
records = []

cols_to_use = [c for c in df.columns if c != "timestamp"]
if FEATURE_WHITELIST is not None:
    cols_to_use = [c for c in cols_to_use if c in FEATURE_WHITELIST]

for col in cols_to_use:
    idx_raw = raw_nan_mask[col].to_numpy()
    if idx_raw.any():
        records.append(pd.DataFrame({
            "timestamp": ts[idx_raw],
            "reason": "raw_nan",
            "feature": col
        }))

    idx_der = derived_nan_mask[col].to_numpy()
    if idx_der.any():
        records.append(pd.DataFrame({
            "timestamp": ts[idx_der],
            "reason": "derived_nan",
            "feature": col
        }))

nan_events = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["timestamp", "reason", "feature"])
nan_events = nan_events.dropna(subset=["timestamp"]).sort_values("timestamp")

# ===== 7) 降采样（按事件行）=====
if EVENT_STRIDE and EVENT_STRIDE > 1 and len(nan_events) > 0:
    nan_events = nan_events.iloc[::EVENT_STRIDE].copy()

print("nan_events rows (after sampling):", len(nan_events))
print(nan_events["reason"].value_counts())

# ===== 8) feature -> 高度：构造 y 类别轴 =====
nan_events["y_cat"] = nan_events["reason"] + " | " + nan_events["feature"]

# 为了让图更整齐，按 feature 排序；并让 raw 在上、derived 在下（可调整）
feat_order = sorted(nan_events["feature"].unique())
cat_order = [f"raw_nan | {f}" for f in feat_order] + [f"derived_nan | {f}" for f in feat_order]
nan_events["y_cat"] = pd.Categorical(nan_events["y_cat"], categories=cat_order, ordered=True)

# ===== 9) 画图：x=时间，y=不同高度（reason|feature）=====
plt.figure(figsize=(14, max(4, 0.22 * len(cat_order))))
plt.scatter(nan_events["timestamp"], nan_events["y_cat"], s=6, alpha=0.5)

plt.xlabel("time")
plt.ylabel("missing reason | feature")
plt.title(f"Missing events over time (feature as height, stride={EVENT_STRIDE})")
plt.tight_layout()
plt.show()