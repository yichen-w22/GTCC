import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
from iapws import IAPWS97

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

df = pd.read_pickle(r"HRSG\jyrd高压余热锅炉_1min.pkl")

# print(df.columns)

cols = pd.Series(df.columns)
df.columns = cols.mask(cols.isna() | (cols == ""), cols.shift())

cols = pd.Series(df.columns)
df.columns = cols + cols.groupby(cols).cumcount().replace(0, "").astype(str).radd(".").replace(".", "")

df = df.replace(0, np.nan)

df["timestamp"] = df.index

高压给水流量 = [
    "#1炉高压给水流量A",
    "#1炉高压给水流量A.1",
    "#1炉高压给水流量B",
    "#1炉高压给水流量B.1",
    "#1炉高压给水流量C",
    "#1炉高压给水流量C.1",
    "#1锅炉高压给水流量"
]
df["高压给水流量"] = df[高压给水流量].mean(axis=1) * 1000 / 3600

df["高压主蒸汽流量"] = df["#1炉高压主蒸汽流量"] * 1000 / 3600

# 高压过热器1入口烟温 = [
#     "#1炉高压过热器1入口烟温1",
#     "#1炉高压过热器1入口烟温2",
#     "#1炉高压过热器1入口烟温3",
#     "#1炉高压过热器1入口烟温4",
#     "#1炉高压过热器1入口烟温5",
#     "#1炉高压过热器1入口烟温6"
# ]
# df["高压过热器1入口烟温"] = df[["#1炉高压过热器1入口烟温1", "#1炉高压过热器1入口烟温4"]].mean(axis=1) + 273.15

高压过热器1入口烟温 = [f"#1炉高压过热器1入口烟温{i}" for i in range(1,7)]
df["高压过热器1入口烟温"] = df[高压过热器1入口烟温].mean(axis=1) + 273.15

高压蒸发器入口烟温 = [f"#1炉高压蒸发器入口烟温{i}" for i in range(1,7)]
df["高压过热器1出口烟温"] = df[高压蒸发器入口烟温].mean(axis=1) + 273.15

rho_g = 1.30   # kg/Nm3
df["烟气流量"] = df["#1炉烟囱出口烟气流量"] * rho_g / 3600


汽包压力 = [
    "#1炉高压汽包压力A",
    "#1炉高压汽包压力B",
    "#1炉高压汽包压力C",
    "#1锅炉高压汽包压力",
]
df["汽包压力"] = df[汽包压力].mean(axis=1)

def calc_sat_temp(p):
    try:
        return IAPWS97(P=p, x=1).T - 273.15   # 转为摄氏度
    except:
        return np.nan

df["高压过热器1入口蒸汽温度"] = df["汽包压力"].apply(calc_sat_temp) + 273.15

df["高压过热器1出口蒸汽温度"] = df["#1炉高压过热汽减温器入口蒸汽温度"] + 273.15

df["汽包压力"] = df["汽包压力"] * 1e6

df_h = df[[
    "timestamp",
    "烟气流量",
    "高压过热器1入口烟温",
    "高压过热器1出口烟温",
    "高压给水流量",
    "高压主蒸汽流量",
    "汽包压力",
    "高压过热器1入口蒸汽温度",
    "高压过热器1出口蒸汽温度"
]]

df_h.to_csv(r"HRSG/jyrd高压余热锅炉过热器1.csv", index=False, encoding="utf-8-sig")

# # 标记每一行是否有NaN
# mask = df_h.notna().all(axis=1)

# # 给连续区段编号
# group = (mask != mask.shift()).cumsum()

# # 找出所有连续无NaN区段
# segments = (
#     df_h[mask]
#     .groupby(group[mask])
# )

# # 找最长区段
# longest_seg = max(segments, key=lambda x: len(x[1]))[1]

# # 最多取10000
# df_h_final = longest_seg.iloc[:10000].copy()

# # 保存
# df_h_final.to_csv(
#     r"HRSG/jyrd高压余热锅炉过热器1_测试数据.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

# plt.plot(df["timestamp"], df["汽包压力"], linewidth=0.8, label="汽包出口蒸汽温度")
# plt.legend()
# plt.show()

# 高压蒸发器入口烟温 = [f"#1炉高压蒸发器入口烟温{i}" for i in range(1,7)]
# df["高压蒸发器入口烟温"] = df[高压蒸发器入口烟温].mean(axis=1)


# start = "2025-07-03 00:00:00"
# end   = "2025-07-05 00:00:00"

# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# start = pd.to_datetime(start)
# end = pd.to_datetime(end)
# mask = df["timestamp"].between(start, end)

# plt.figure(figsize=(10, 5))
# for c in 汽包压力:
#     plt.plot(df["timestamp"],
#              df[c],
#              linewidth=0.8,
#              label=c)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # 计算逐行均值与标准差
# df["mean"] = df[汽包压力].mean(axis=1, skipna=True)
# df["std"]  = df[汽包压力].std(axis=1, skipna=True)

# plt.figure(figsize=(10, 5))
# # plt.plot(df["timestamp"], df["高压给水流量_mean"], linewidth=0.8, label="mean")
# plt.plot(df["timestamp"], df["std"], linewidth=0.8, label="std")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 5))

# for c in 高压给水流量:
#     plt.plot(df["timestamp"], df[c], linewidth=0.8, label=c)

# plt.legend()
# plt.tight_layout()
# plt.show()


# start = "2025-07-01 00:00:00"
# end   = "2025-07-15 00:00:00"

# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# start = pd.to_datetime(start)
# end = pd.to_datetime(end)
# mask = df["timestamp"].between(start, end)

# plt.figure(figsize=(10, 5))

# cols_temp = ["高压过热汽减温器入口蒸汽温度", "高压过热器1入口烟温"]

# for c in cols_temp:
#     plt.plot(df.loc[mask, "timestamp"],
#              df.loc[mask, c],
#              linewidth=0.8,
#              label=c)

# plt.xlabel("time")
# plt.ylabel("Temperature")
# plt.title("高压过热汽减温器入口蒸汽温度 和 高压过热器1入口烟温 随时间变化")
# plt.legend()
# plt.tight_layout()
# plt.show()
