import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\temp\plant_metrics.csv")

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# 高压缸
mask_hp = df["ST.高压缸出力"] > 0
axes[0].plot(df["ST.汽轮机功率"][mask_hp] / 1e6, df["ST.高压缸等熵效率"][mask_hp], ".", markersize=2, alpha=0.5)
axes[0].set_xlabel("汽轮机功率 / MW")
axes[0].set_ylabel("高压缸等熵效率")
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.5, 1.0)

# 中压缸
mask_ip = df["ST.中压缸出力"] > 0
axes[1].plot(df["ST.汽轮机功率"][mask_ip] / 1e6, df["ST.中压缸等熵效率"][mask_ip], ".", markersize=2, alpha=0.5)
axes[1].set_xlabel("汽轮机功率 / MW")
axes[1].set_ylabel("中压缸等熵效率")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0.5, 1.0)

# 低压缸
mask_lp = df["ST.低压缸出力（正算）"] > 0
axes[2].plot(df["ST.汽轮机功率"][mask_lp] / 1e6, df["ST.低压缸等熵效率"][mask_lp], ".", markersize=2, alpha=0.5)
axes[2].set_xlabel("汽轮机功率 / MW")
axes[2].set_ylabel("低压缸等熵效率")
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0.5, 1.0)

plt.tight_layout()
plt.show()