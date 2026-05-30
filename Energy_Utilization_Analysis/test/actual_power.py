import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\temp\plant_metrics.csv")
raw_df = pd.read_csv(r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\averaged_data_10min.csv")

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

XLABELS1 = "GT1.燃机实际功"
XLABELS2 = "GT2.燃机实际功"


YLABELS1 = "GT1.燃机计算功"
YLABELS2 = "GT2.燃机计算功"
TEMP_LABELS1 = "环境温度_1"
TEMP_LABELS2 = "环境温度_2"

# mask_both_running = (
#     (df["GT1.是否运行"] == True)
#     & (df["GT2.是否运行"] == True)
# )

# mask1 = (df[YLABELS1] > 0) & mask_both_running
# mask2 = (df[YLABELS2] > 0) & mask_both_running

upper_bound = 300 * 1e6
lower_bound = 0

mask1 = (df[YLABELS1] > lower_bound) & (df[YLABELS1] < upper_bound)
mask2 = (df[YLABELS2] > lower_bound) & (df[YLABELS2] < upper_bound)

idx = df["idx"].astype(int)
temp1 = raw_df.loc[idx, TEMP_LABELS1].reset_index(drop=True)
temp2 = raw_df.loc[idx, TEMP_LABELS2].reset_index(drop=True)
if temp1.median() > 200:
    temp1 = temp1 - 273.15
if temp2.median() > 200:
    temp2 = temp2 - 273.15

vmin = min(temp1[mask1].quantile(0.01), temp2[mask2].quantile(0.01))
vmax = max(temp1[mask1].quantile(0.99), temp2[mask2].quantile(0.99))

scatter0 = axes[0].scatter(
    df[XLABELS1][mask1] / 1e6,
    df[YLABELS1][mask1] / 1e6 - 5,
    c=temp1[mask1],
    cmap="coolwarm",
    vmin=vmin,
    vmax=vmax,
    s=4,
    alpha=0.45,
    edgecolors="none",
)
axes[0].plot([lower_bound / 1e6, upper_bound / 1e6], [lower_bound / 1e6, upper_bound / 1e6], "r--", linewidth=1.2)
axes[0].set_xlabel("GT1 实际功率 / MW")
axes[0].set_ylabel("GT1 计算功率 / MW")
axes[0].grid(True, alpha=0.3)


axes[1].scatter(
    df[XLABELS2][mask2] / 1e6,
    df[YLABELS2][mask2] / 1e6 - 5,
    c=temp2[mask2],
    cmap="coolwarm",
    vmin=vmin,
    vmax=vmax,
    s=4,
    alpha=0.45,
    edgecolors="none",
)
axes[1].plot([lower_bound / 1e6, upper_bound / 1e6], [lower_bound / 1e6, upper_bound / 1e6], "r--", linewidth=1.2)
# axes[1].plot(df[XLABELS1][mask1], df[YLABELS1][mask1], ".", markersize=2, label="残差")
axes[1].set_xlabel("GT2 实际功率 / MW")
axes[1].set_ylabel("GT2 计算功率 / MW")
axes[1].grid(True, alpha=0.3)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.72])
cbar = fig.colorbar(scatter0, cax=cbar_ax)
cbar.set_label("环境温度 / ℃")

fig.subplots_adjust(wspace=0.12, right=0.88)
plt.show()
