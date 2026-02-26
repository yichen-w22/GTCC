import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r"data_processing\for_degradation\outcome\jqrd_compressor.csv")

def steady_state_index(
    series: pd.Series,
    time_dt: pd.Series,
    window: int = 10,
    std_quantile: float = 0.9,
    slope_quantile: float = 0.9,
    power_low: float = 100,
    power_high: float = 300,
):
    # 这个程序用于筛选功率的稳定工况点
    # 只保留必要列，保留原 index
    tmp = pd.DataFrame({
        "value": pd.to_numeric(series, errors="coerce"),
        "time": pd.to_datetime(time_dt, errors="coerce")
    }).dropna(subset=["value", "time"])

    v = tmp["value"]

    # rolling 统计量
    rolling_std = v.rolling(window=window, center=True).std()
    rolling_slope = v.diff().rolling(window=window, center=True).mean().abs()

    # 自适应阈值（分位数）
    std_thr = rolling_std.quantile(std_quantile)
    slope_thr = rolling_slope.quantile(slope_quantile)

    # 稳态判据
    steady_mask = (rolling_std < std_thr) & (rolling_slope < slope_thr)

    # 功率区间筛选
    if power_low is not None:
        steady_mask &= (v >= power_low)
    if power_high is not None:
        steady_mask &= (v <= power_high)

    # 返回原 df 的 index
    steady_index = tmp.index[steady_mask.fillna(False)]

    return steady_index, std_thr, slope_thr

print(df.shape)

df = df.loc[steady_state_index(df["unit_power_MW"], pd.to_datetime(df["timestamp"]))[0]]

print(df.shape)

t0 = df["timestamp"].iloc[0]
t_sec = (df["timestamp"] - t0).dt.total_seconds()

plt.figure(figsize=(12,4), dpi=150)
plt.scatter(t_sec, df["unit_power_MW"])
plt.xlabel("时间")
plt.ylabel("机组功率 (MW)")
plt.title("机组功率稳态工况筛选结果")
plt.tight_layout()
plt.show()
print("图已保存为 steady_power.png")