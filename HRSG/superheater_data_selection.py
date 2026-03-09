import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib
from scipy.signal import butter, filtfilt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# 读取数据
df = pd.read_csv(r"HRSG\jyrd高压余热锅炉过热器1.csv")


cols = ["高压过热器1入口烟温",
    "高压过热器1出口烟温",
    "高压过热器1入口蒸汽温度",
    "高压过热器1出口蒸汽温度",
    "烟气流量",
    "高压给水流量",
    "高压主蒸汽流量"
    ]

# 滑窗长度：例如 5 个点 = 5 分钟
window = 5
df_filtered = df.copy()

for col in cols:
    df_filtered[col] = (
        pd.to_numeric(df_filtered[col], errors="coerce")
        .rolling(window=window, center=True)
        .mean()
        .bfill()
        .ffill()
    )

for col in cols:
    df_filtered[col] = (
        pd.to_numeric(df_filtered[col], errors="coerce")
        .rolling(window=window, center=True)
        .mean()
        .bfill()
        .ffill()
    )

# for col in cols:
#     df_filtered[col] = (
#         pd.to_numeric(df_filtered[col], errors="coerce")
#         .rolling(window=window, center=True)
#         .mean()
#         .bfill()
#         .ffill()
#     )

df_filtered = df_filtered[27500:30000]

df_filtered.to_csv(r"HRSG\jyrd高压余热锅炉过热器1_滑窗平均数据1.csv", index=False)

# # 示例对比绘图
# plt.figure(figsize=(12, 6))
# plt.plot(df["烟气流量"], alpha=0.5)
# plt.plot(df_filtered["烟气流量"], linewidth=2)
# plt.xlabel("时间")
# plt.ylabel("温度")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()



# # 采样频率
# fs = 1 / 60      # Hz (1分钟采样)

# # 截止频率（例如10分钟周期）
# cutoff = 1 / (10*60)

# # 设计滤波器
# b, a = butter(N=10, Wn=cutoff/(fs/2), btype='low')

# df_filtered = df.copy()


# columns = ["高压过热器1入口烟温",
#            "高压过热器1出口烟温",
#            "高压过热器1入口蒸汽温度",
#            "高压过热器1出口蒸汽温度",
#            "烟气流量",
#            "高压给水流量",
#            "高压主蒸汽流量"
#            ]


# for col in columns:
#     df_filtered[col] = filtfilt(b, a, df[col])

# df_filtered.to_csv(r"HRSG\jyrd高压余热锅炉过热器1_滤波后数据.csv", index=False)

fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2*len(cols)), sharex=True)

for i, col in enumerate(cols):
    axes[i].plot(df_filtered[col])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# fig, axes = plt.subplots(len(columns), 1, figsize=(12, 2*len(columns)), sharex=True)

# for i, col in enumerate(columns):
#     axes[i].plot(df_filtered[col])
#     axes[i].set_title(col)

# plt.tight_layout()
# plt.show()



# from scipy.signal import correlate
# import numpy as np

# Tg = df["烟气流量"].values
# Ts = df["高压主蒸汽流量"].values

# Tg = Tg - Tg.mean()
# Ts = Ts - Ts.mean()

# corr = correlate(Ts, Tg, mode="full")

# lags = np.arange(-len(Tg)+1, len(Tg))

# delay = lags[np.argmax(corr)]

# print("响应延迟:", delay, "个采样点")
# print("响应时间:", delay, "分钟")