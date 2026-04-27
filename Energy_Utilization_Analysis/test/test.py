import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\energy_analysis\GT_model\result\gt2_results_wide.csv"
)

window = 20
peak_min_distance = 50
peak_sigma = 1.5


def find_peak_indices(series, threshold, min_distance):
    values = series.to_numpy()
    candidates = []

    for i in range(1, len(values) - 1):
        prev_value = values[i - 1]
        curr_value = values[i]
        next_value = values[i + 1]

        is_positive_peak = curr_value >= prev_value and curr_value > next_value and curr_value >= threshold
        is_negative_peak = curr_value <= prev_value and curr_value < next_value and curr_value <= -threshold

        if is_positive_peak or is_negative_peak:
            candidates.append(i)

    selected = []
    for idx in sorted(candidates, key=lambda item: abs(values[item]), reverse=True):
        if all(abs(idx - kept) >= min_distance for kept in selected):
            selected.append(idx)

    return sorted(selected)


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 计算
df["residual"] = df["net_power"] - df["actual_power"]
df["air_fuel_ratio"] = df["state_1_m_dot"] / df["fuel_m_dot"]

# 计算滑窗均值
df["residual_ma"] = df["residual"].rolling(window=window, center=True, min_periods=1).mean()
df["air_fuel_ratio_ma"] = df["air_fuel_ratio"].rolling(window=window, center=True, min_periods=1).mean()
df["state_1_m_dot_ma"] = df["state_1_m_dot"].rolling(window=window, center=True, min_periods=1).mean()
df["fuel_m_dot_ma"] = df["fuel_m_dot"].rolling(window=window, center=True, min_periods=1).mean()
df["actual_power_ma"] = df["actual_power"].rolling(window=window, center=True, min_periods=1).mean()
df["compressor_efficiency_ma"] = df["compressor_efficiency"].rolling(window=window, center=True, min_periods=1).mean()
df["turbine_efficiency_ma"] = df["turbine_efficiency"].rolling(window=window, center=True, min_periods=1).mean()

# 基于平滑后的残差识别主要正峰和负峰
peak_threshold = peak_sigma * df["residual_ma"].std()
peak_indices = find_peak_indices(df["residual_ma"], peak_threshold, peak_min_distance)
peak_df = df.iloc[peak_indices].copy()
positive_peaks = peak_df[peak_df["residual_ma"] >= 0]
negative_peaks = peak_df[peak_df["residual_ma"] < 0]

# 创建子图
fig, axes = plt.subplots(7, 1, figsize=(12, 14), dpi=150, sharex=True)

# ---- 子图1：功率残差 ----
axes[0].plot(df["source_idx"], df["residual"], ".", markersize=2, label="残差")
axes[0].plot(df["source_idx"], df["residual_ma"], color="red", linewidth=1.5, label="滑窗均值")
axes[0].set_ylabel("功率残差 (MW)")
axes[0].set_title("机组功率残差")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper right")

for _, row in positive_peaks.iterrows():
    axes[0].axvline(row["source_idx"], color="crimson", linestyle="--", linewidth=0.8, alpha=0.7)
for _, row in negative_peaks.iterrows():
    axes[0].axvline(row["source_idx"], color="royalblue", linestyle="--", linewidth=0.8, alpha=0.7)

# ---- 子图2：空燃比 ----
axes[1].plot(df["source_idx"], df["air_fuel_ratio"], ".", markersize=2)
axes[1].plot(df["source_idx"], df["air_fuel_ratio_ma"], color="red", linewidth=1.5)
axes[1].set_ylabel("空燃比")
axes[1].set_title("空燃比变化")
axes[1].grid(True, alpha=0.3)

# ---- 子图3：空气流量 ----
axes[2].plot(df["source_idx"], df["state_1_m_dot"], ".", markersize=2)
axes[2].plot(df["source_idx"], df["state_1_m_dot_ma"], color="red", linewidth=1.5)
axes[2].set_ylabel("空气流量")
axes[2].set_title("空气流量")
axes[2].grid(True, alpha=0.3)

# ---- 子图4：燃料流量 ----
axes[3].plot(df["source_idx"], df["fuel_m_dot"], ".", markersize=2)
axes[3].plot(df["source_idx"], df["fuel_m_dot_ma"], color="red", linewidth=1.5)
axes[3].set_ylabel("燃料流量")
axes[3].set_title("燃料流量")
axes[3].grid(True, alpha=0.3)

# ---- 子图5：输出功 ----
axes[4].plot(df["source_idx"], df["actual_power"], ".", markersize=2)
axes[4].plot(df["source_idx"], df["actual_power_ma"], color="red", linewidth=1.5)
axes[4].set_ylabel("actual_power")
axes[4].set_title("actual_power")
axes[4].grid(True, alpha=0.3)

# ---- 子图6：压气机效率 ----
axes[5].plot(df["source_idx"], df["compressor_efficiency"], ".", markersize=2)
axes[5].plot(df["source_idx"], df["compressor_efficiency_ma"], color="red", linewidth=1.5)
axes[5].set_ylabel("压气机效率")
axes[5].set_title("压气机效率")
axes[5].grid(True, alpha=0.3)

# ---- 子图7：透平效率 ----
axes[6].plot(df["source_idx"], df["turbine_efficiency"], ".", markersize=2)
axes[6].plot(df["source_idx"], df["turbine_efficiency_ma"], color="red", linewidth=1.5)
axes[6].set_xlabel("时间")
axes[6].set_ylabel("透平效率")
axes[6].set_title("透平效率")
axes[6].grid(True, alpha=0.3)

# 在后续子图中标出功率残差峰值对应时刻
for ax in axes[1:]:
    for _, row in positive_peaks.iterrows():
        ax.axvline(row["source_idx"], color="crimson", linestyle="--", linewidth=0.8, alpha=0.7)
    for _, row in negative_peaks.iterrows():
        ax.axvline(row["source_idx"], color="royalblue", linestyle="--", linewidth=0.8, alpha=0.7)

plt.tight_layout()
plt.show()
