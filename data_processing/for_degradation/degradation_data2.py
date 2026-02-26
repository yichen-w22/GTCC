# 这是包含了数据质量分析的生成程序

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

EVENT_STRIDE = 1000
FEATURE_WHITELIST = [
    "comp_inlet_T_K", "comp_inlet_P_MPa",
    "comp_exit_T_K", "comp_exit_P_MPa",
    "IGV_deg", "eta_isentropic",
    "shaft_rpm"
]

BASE_DIR = Path(__file__).resolve().parent.parent.parent
p = BASE_DIR / Path(r"datareader_new\jqrd\outcome\jqrd燃机1_1min.pkl")

df_raw = pd.read_pickle(p)
cols = pd.Series(df_raw.columns)
df_raw.columns = cols + cols.groupby(cols).cumcount().replace(0, "").astype(str).radd(".").replace(".", "")

df = pd.DataFrame(index=df_raw.index)
df["timestamp"] = df_raw.index

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

# unit_power_MW
df["unit_power_MW"] = df_raw["APPARENT POWER"]

df = df.where(df["shaft_rpm"] > 2950)

# ===== raw_nan：插值/派生计算前的缺失 =====
raw_nan_mask = df.isna().copy()



# 统计每行中nan出现的数量分布
# nan_per_row = df.isna().sum(axis=1)

# ratio_by_k = nan_per_row.value_counts(normalize=True).sort_index()

# print(ratio_by_k)


# 统计各列中连续nan出现数量的频率
# import pandas as pd

# def count_consecutive_nan_1d(s: pd.Series, max_k=3):
#     """
#     统计一列中连续 NaN 段的长度分布
#     返回：dict {1: count, 2: count, 3: count}
#     """
#     is_nan = s.isna()

#     # 给连续段编号（NaN / 非 NaN 都会编号）
#     grp = (is_nan != is_nan.shift()).cumsum()

#     # 只保留 NaN 段
#     nan_groups = is_nan.groupby(grp).sum()
#     nan_groups = nan_groups[nan_groups > 0]

#     # 统计长度
#     out = {k: 0 for k in range(1, max_k + 1)}
#     for k, v in nan_groups.value_counts().items():
#         if k <= max_k:
#             out[k] = v

#     return out

# def count_consecutive_nan_by_column(df: pd.DataFrame, max_k=3):
#     records = []

#     for col in df.columns:
#         res = count_consecutive_nan_1d(df[col], max_k=max_k)
#         for k, cnt in res.items():
#             records.append({
#                 "feature": col,
#                 "nan_run_length": k,
#                 "count": cnt
#             })

#     return pd.DataFrame(records)

# nan_run_stats = count_consecutive_nan_by_column(df, max_k=20)
# nan_run_stats.to_csv("temp/nan_run_stats.csv", index=False)


# # ===== 插值（只对关键输入列）=====
# cols_interp = ["comp_inlet_T_K", "comp_inlet_P_MPa", "comp_exit_T_K", "comp_exit_P_MPa", "IGV_deg"]
# df[cols_interp] = df[cols_interp].apply(
#     lambda s: s.interpolate(method="linear", limit=9, limit_area="inside")
# )

# ===== 1. 插值前保存一份 =====
df_before = df.copy()

# ===== 2. 你的插值代码 =====
cols_interp1 = [
    "comp_exit_T_K",
    "comp_exit_P_MPa",
    "IGV_deg",
    "unit_power_MW"
]

df[cols_interp1] = df[cols_interp1].apply(
    lambda s: s.interpolate(
        method="linear",
        limit=9,
        limit_area="inside"
    )
)

cols_interp2 = [
    "comp_inlet_T_K",
    "comp_inlet_P_MPa",
]

df[cols_interp2] = df[cols_interp2].apply(
    lambda s: s.interpolate(
        method="linear",
        limit=30,
        limit_area="inside"
    )
)

df_after = df.copy()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_interp_points(df_raw, df_interp, col, ts_col="timestamp"):
    t = pd.to_datetime(df_interp[ts_col])

    s_raw = df_raw[col]
    s_itp = df_interp[col]

    mask_raw = ~s_raw.isna()
    mask_interp = s_raw.isna() & ~s_itp.isna()
    mask_nan = s_raw.isna() & s_itp.isna()

    n_total = len(s_raw)
    n_raw = mask_raw.sum()
    n_itp = mask_interp.sum()
    n_nan = mask_nan.sum()

    # 比例
    r_raw = n_raw / n_total
    r_itp = n_itp / n_total
    r_nan = n_nan / n_total

    plt.figure(figsize=(12, 4))

    # ===== 原有点 =====
    plt.scatter(
        t[mask_raw],
        s_raw[mask_raw],
        s=1,
        alpha=0.25,
        label=f"raw ({r_raw:.1%})"
    )

    # ===== 插值点 =====
    plt.scatter(
        t[mask_interp],
        s_itp[mask_interp],
        s=1,
        alpha=0.25,
        label=f"interpolated ({r_itp:.1%})"
    )

    # ===== 仍为空值 =====
    if n_nan > 0:
        # 放在当前数据最小值略下方，避免干扰
        y_ref = s_itp.dropna().min()
        y_nan = np.full(n_nan, y_ref)

        plt.scatter(
            t[mask_nan],
            y_nan,
            s=1,
            alpha=0.25,
            marker="x",
            label=f"still NaN ({r_nan:.1%})"
        )

    plt.xlabel("time")
    plt.ylabel(col)
    plt.title(f"Interpolation audit: {col}")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()


# ===== 4. 对每个特征分别画图 =====
# for col in cols_interp1:
#     plot_interp_points(df_before, df_after, col)
    
for col in cols_interp2:
    plot_interp_points(df_before, df_after, col)

# n_nan_rows = df.isna().any(axis=1).sum()
# print(n_nan_rows)
# ratio = df.isna().any(axis=1).mean()
# print(f"存在 NaN 的行占比: {ratio:.2%}")

# ===== 等熵效率（可能产生派生缺失）=====
gamma = 1.385
T1 = df["comp_inlet_T_K"]
T2 = df["comp_exit_T_K"]
P1 = df["comp_inlet_P_MPa"]
P2 = df["comp_exit_P_MPa"]

PR = P2 / P1
T2s = T1 * (PR ** ((gamma - 1.0) / gamma))
df["eta_isentropic"] = (T2s - T1) / (T2 - T1)

# # ===== derived_nan：插值/计算后新出现的缺失 =====
# pre_drop_nan_mask = df.isna().copy()
# raw_nan_mask = raw_nan_mask.reindex(columns=df.columns, fill_value=False)
# derived_nan_mask = pre_drop_nan_mask & (~raw_nan_mask)

# # ===== filtered_rpm：转速筛选导致的删除（不是 NaN）=====
# rpm_fail_mask = df["shaft_rpm"] <= 2950

# # ===== filtered_dropna：最终 dropna 会删掉的行（仍有任意 NaN）=====
# dropna_fail_mask = df.isna().any(axis=1)

# # ===== 构造事件表（timestamp, reason, feature）=====
# ts = pd.to_datetime(df["timestamp"], errors="coerce")
# records = []

# cols_to_use = [c for c in df.columns if c != "timestamp"]
# if FEATURE_WHITELIST is not None:
#     cols_to_use = [c for c in cols_to_use if c in FEATURE_WHITELIST]

# raw_nan_mask = raw_nan_mask.reindex(columns=df.columns, fill_value=False)
# derived_nan_mask = derived_nan_mask.reindex(columns=df.columns, fill_value=False)

# for col in cols_to_use:
#     idx_raw = raw_nan_mask[col].to_numpy()
#     if idx_raw.any():
#         records.append(pd.DataFrame({"timestamp": ts[idx_raw], "reason": "raw_nan", "feature": col}))

#     idx_der = derived_nan_mask[col].to_numpy()
#     if idx_der.any():
#         records.append(pd.DataFrame({"timestamp": ts[idx_der], "reason": "derived_nan", "feature": col}))

# # rpm / dropna 事件（按“规则”统计，不按单列）
# records.append(pd.DataFrame({
#     "timestamp": ts[rpm_fail_mask.to_numpy()],
#     "reason": "filtered_rpm",
#     "feature": "shaft_rpm"
# }))
# records.append(pd.DataFrame({
#     "timestamp": ts[dropna_fail_mask.to_numpy()],
#     "reason": "filtered_dropna",
#     "feature": "any"
# }))

# events = pd.concat(records, ignore_index=True)
# events = events.dropna(subset=["timestamp"]).sort_values("timestamp")

# # ===== 降采样 =====
# if EVENT_STRIDE and EVENT_STRIDE > 1 and len(events) > 0:
#     events = events.iloc[::EVENT_STRIDE].copy()

# print("events rows (after sampling):", len(events))
# print(events["reason"].value_counts())

# # ===== y 轴：reason|feature 映射为高度 =====
# events["y_cat"] = events["reason"] + " | " + events["feature"]

# feat_order = sorted(events["feature"].unique())
# reason_order = ["raw_nan", "derived_nan", "filtered_rpm", "filtered_dropna"]
# cat_order = []
# for r in reason_order:
#     for f in feat_order:
#         k = f"{r} | {f}"
#         if k in set(events["y_cat"]):
#             cat_order.append(k)

# events["y_cat"] = pd.Categorical(events["y_cat"], categories=cat_order, ordered=True)

# plt.figure(figsize=(14, max(4, 0.22 * len(cat_order))))
# plt.scatter(events["timestamp"], events["y_cat"], s=6, alpha=0.5)
# plt.xlabel("time")
# plt.ylabel("reason | feature")
# plt.title(f"Missing & filtered events over time (stride={EVENT_STRIDE})")
# plt.tight_layout()
# plt.show()

# ===== 最后才做清洗数据输出 =====
df_clean = df.dropna()
df_clean.to_csv(
    r"data_processing\for_degradation\outcome\jqrd_compressor_degradation2.csv",
    encoding="utf-8-sig",
    index=False
)