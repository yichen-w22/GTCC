import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# def steady_state_index_fast(series: pd.Series,
#                             time_dt: pd.Series,
#                             window: int = 10,
#                             std_quantile: float = 0.9,
#                             slope_quantile: float = 0.9,
#                             power_low: float = 100,
#                             power_high: float = 300,
#                             assume_sorted: bool = False):
#     # 只保留必要列 + 保留原 index
#     tmp = pd.DataFrame({"value": pd.to_numeric(series, errors="coerce"),
#                         "time": time_dt}).dropna(subset=["value", "time"])

#     # 如果不确定是否按时间排序，才 sort
#     if not assume_sorted:
#         # 如果已单调递增则不 sort（省很多时间）
#         if not tmp["time"].is_monotonic_increasing:
#             tmp = tmp.sort_values("time")

#     v = tmp["value"]  # Series (float)
#     # rolling 统计
#     rolling_std = v.rolling(window=window, center=True).std()
#     rolling_slope = v.diff().rolling(window=window, center=True).mean().abs()

#     std_thr = rolling_std.quantile(std_quantile)
#     slope_thr = rolling_slope.quantile(slope_quantile)

#     steady_mask = (rolling_std < std_thr) & (rolling_slope < slope_thr)
#     power_mask = (v > power_low) & (v < power_high)

#     mask = steady_mask & power_mask
#     return tmp.index[mask], std_thr, slope_thr

def remove_outlier_df(df, cols, n=3):
    df = df.copy()
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")  # 保证可算
    for c in cols:
        x = df[c]
        mu, sigma = x.mean(skipna=True), x.std(skipna=True)
        df[c] = x.where(x.between(mu - n*sigma, mu + n*sigma))  # 离群点 -> NaN
    return df

# ===== 主流程：timestamp 只转一次 =====
df = pd.read_csv(r"data_processing\for_degradation\outcome\jqrd_compressor.csv")

# df = df.dropna()
df["comp_inlet_P_MPa"] = df["comp_inlet_P_MPa"].where((df["comp_inlet_P_MPa"] > 0) & (df["comp_inlet_P_MPa"] < 1))
# df["comp_inlet_P_MPa"] = df["comp_inlet_P_MPa"].interpolate(method="linear", limit=2)
# df = df.dropna()
df["shaft_rpm"] = df["shaft_rpm"].where((df["shaft_rpm"] > 2950) & (df["shaft_rpm"] < 3050))
df["unit_power_MW"] = df["unit_power_MW"].where((df["unit_power_MW"] > 100) & (df["unit_power_MW"] < 300))


df["air_inlet_flow_kg_s"] = df["air_inlet_flow_kg_s"].where(df["air_inlet_flow_kg_s"] > 1)
df["air_inlet_flow_kg_s"] = df["air_inlet_flow_kg_s"].interpolate(method="linear", limit=1)
print(df["air_inlet_flow_kg_s"].isna().sum())

df["comp_inlet_T_K"] = df["comp_inlet_T_K"].where(df["comp_inlet_T_K"] > 0)
# df = df.dropna()

df["Wc"] = df["air_inlet_flow_kg_s"] * np.sqrt(df["comp_inlet_T_K"] / 288.15) / (df["comp_inlet_P_MPa"] / 0.101325)
# print(df["Wc"])
# mu, sigma = df["Wc"].mean(skipna=True), df["Wc"].std(skipna=True)
# print("Wc mean:", mu, " std:", sigma)


df["Nc"] = df["shaft_rpm"] * np.sqrt(288.15 / df["comp_inlet_T_K"])

df = remove_outlier_df(df, ["Wc"], n=3)
df = remove_outlier_df(df, ["Nc"], n=3)

# idx = df.loc[(df["Nc"] > 3000) & (df["Nc"] < 3050)].index
# print(df.loc[idx])
# idx = df.loc[(df["Nc"] > 3000) & (df["Nc"] < 3050)].index
# print(df.loc[idx])

# print((df["Nc"].isna() & ~df["Wc"].isna()).sum())
# print((~df["Nc"].isna() & df["Wc"].isna()).sum())
# print((df["Nc"].isna() & df["Wc"].isna()).sum())
# print(df["Nc"].isna().sum())
# print(df["Wc"].isna().sum())
# print(df["comp_inlet_P_MPa"].isna().sum())
# print(df["comp_inlet_T_K"].isna().sum())
# print(df["air_inlet_flow_kg_s"].isna().sum())

plt.figure(figsize=(12, 4))
plt.plot(df["Wc"], df["Nc"], ".")  # 通常比 scatter 快
# plt.plot(df["timestamp"], df["Nc"], ".")
# plt.plot(df["timestamp"], df["Wc"], ".")
# df = df.sample(frac=0.1, random_state=42)  # 只画 10% 点，防止过密
# plt.plot(df["timestamp"], df["air_inlet_flow_kg_s"])
plt.xlabel("折合流量 Wc")
plt.ylabel("折合转速 Nc")
plt.title("压气机 Wc–Nc 分布")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# # print("Wc non-NaN:", df["Wc"].notna().sum(), " / ", len(df))
# # print("Nc non-NaN:", df["Nc"].notna().sum(), " / ", len(df))
# # print("both non-NaN:", df[["Wc","Nc"]].notna().all(axis=1).sum())

# # print("Wc inf:", np.isinf(df["Wc"]).sum(), "Nc inf:", np.isinf(df["Nc"]).sum())
# # print("P<=0 count:", (pd.to_numeric(df["comp_inlet_P_MPa"], errors="coerce") <= 0).sum())


# start = "2024-06-01 00:00:00"
# end   = "2024-07-01 00:00:00"

# # 只取你指定的时间段
# df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

# plt.figure(figsize=(12, 4))
# plt.plot(df["timestamp"], df["air_inlet_flow_kg_s"])
# plt.xlabel("时间")
# plt.ylabel("进气流量 (kg/s)")
# plt.title("压气机进气流量随时间变化")
# plt.grid(True)
# plt.show()

