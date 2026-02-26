import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


# ===== 参数区（只改这里） =====
CSV_PATH = r"data_processing\for_degradation\outcome\jqrd_compressor_degradation2.csv"

Y = "eta_isentropic"
X = ["comp_inlet_T_K", "comp_inlet_P_MPa", "IGV_deg", "unit_power_MW"]

K = 20          # 时间等分段数
DEGREE = 2
ALPHA = 1.0
# ===============================


# 1. 读数据
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

d = (
    df[X + [Y, "timestamp"]]
    .dropna()
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# 2. 时间等分（按样本数）
d["seg"] = pd.cut(
    np.arange(len(d)),
    bins=K,
    labels=False,
    include_lowest=True
)

# 3. 每个 bin 的时间中位数（作为横坐标）
t_mid = (
    d.groupby("seg")["timestamp"]
     .median()
     .sort_index()
)
x_time = t_mid.values   # shape: (K,)

# 4. 交叉段残差矩阵
mat = np.full((K, K), np.nan)

for i in range(K):
    train = d[d["seg"] == i]

    model = make_pipeline(
        PolynomialFeatures(DEGREE, include_bias=False),
        Ridge(alpha=ALPHA)
    )
    model.fit(train[X].to_numpy(), train[Y].to_numpy())

    y_hat = model.predict(d[X].to_numpy())
    delta = d[Y].to_numpy() - y_hat

    for j in range(K):
        # mat[i, j] = np.nanmedian(delta[d["seg"] == j])
        mat[i, j] = np.mean(delta[d["seg"] == j])

# 5. 平均交叉退化曲线（对所有 train bin 求平均）
mean_curve = np.nanmean(mat, axis=0)   # shape: (K,)


# ================== 可视化 ==================

# ---- (A) 每个 bin 一张小图，拼成大图 ----
ncols = int(np.ceil(np.sqrt(K)))
nrows = int(np.ceil(K / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4 * ncols, 3 * nrows),
    sharex=True,
    sharey=True
)

axes = np.asarray(axes).ravel()

for i in range(K):
    ax = axes[i]
    ax.plot(x_time, mat[i], marker="o", alpha=0.8)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"bin {i}", fontsize=10)

# 删除多余子图
for j in range(K, len(axes)):
    fig.delaxes(axes[j])

# 时间轴格式
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
for ax in axes[:K]:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

fig.text(0.5, 0.04, "time (median of each bin)", ha="center")
fig.text(0.04, 0.5, "median residual (eta - eta_hat)", va="center", rotation="vertical")

fig.suptitle("Cross-segment residual lines (each train bin)", fontsize=14)
fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.show()


# ---- (B) 平均交叉退化曲线（核心总结图） ----
plt.figure(figsize=(9, 4.6))
plt.plot(x_time, mean_curve, marker="o", linewidth=2)
plt.axhline(0, linewidth=1)

plt.xlabel("time (median of each bin)")
plt.ylabel("mean cross residual (eta - eta_hat)")
plt.title("Mean cross-segment degradation curve")

plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()