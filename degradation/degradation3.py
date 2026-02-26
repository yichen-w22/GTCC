import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


# ===== 参数（你只需要改这里） =====
CSV_PATH = r"data_processing\for_degradation\outcome\jqrd_compressor_degradation.csv"

Y = "eta_isentropic"
X = ["comp_inlet_T_K", "comp_inlet_P_MPa", "IGV_deg", "unit_power_MW"]

K = 6          # 时间等分段数
DEGREE = 2
ALPHA = 1.0
# ===================================


# 1. 读数据
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

d = df[X + [Y, "timestamp"]].dropna().sort_values("timestamp").reset_index(drop=True)

# 2. 时间等分（按样本数）
d["seg"] = pd.cut(np.arange(len(d)), bins=K, labels=False, include_lowest=True)

# 3. 交叉段残差矩阵（核心）
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
        mat[i, j] = np.nanmedian(delta[d["seg"] == j])

# 自动决定行列数（尽量接近正方形）
ncols = int(np.ceil(np.sqrt(K)))
nrows = int(np.ceil(K / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4 * ncols, 3 * nrows),
    sharex=True, sharey=True
)

axes = np.asarray(axes).ravel()  # 拉平成一维，方便索引

x = np.arange(K)

for i in range(K):
    ax = axes[i]
    ax.plot(x, mat[i], marker="o", alpha=0.8)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"bin {i}", fontsize=10)

# 多余的子图删掉（当 K 不是满格时）
for j in range(K, len(axes)):
    fig.delaxes(axes[j])

# 统一坐标轴标签
fig.text(0.5, 0.04, "test segment (time order)", ha="center")
fig.text(0.04, 0.5, "median residual (eta - eta_hat)", va="center", rotation="vertical")

fig.suptitle("Cross-segment residual lines", fontsize=14)
fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.show()