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

# 创建图
fig, ax = plt.subplots(figsize=(6, 6))

xlabel = "ST.汽轮机功率"
ylabel = "ST.汽轮机计算功率"

upper_bound = 150 * 1e6
lower_bound = 0

mask = (df[ylabel] > lower_bound) & (df[ylabel] < upper_bound)

ax.plot(df[xlabel][mask] / 1e6, (df[ylabel][mask] / 1e6 - 3) , ".", markersize=2, alpha=0.9)
ax.plot([lower_bound / 1e6, upper_bound / 1e6], [lower_bound / 1e6, upper_bound / 1e6], "r--", linewidth=1.2)
ax.set_xlabel("汽轮机实际功率 / MW")
ax.set_ylabel("汽轮机计算功率 / MW")
ax.set_xlim(40, 150)
ax.set_ylim(40, 150)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()