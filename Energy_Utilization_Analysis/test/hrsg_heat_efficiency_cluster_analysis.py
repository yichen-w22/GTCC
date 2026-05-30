from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

KEEP_DENSITY_RATIO = 0.25
N_CLUSTERS = 3
K_NEIGHBORS = 20
POINT_SIZE = 8
POINT_ALPHA = 0.65

UNIT_COLS = {
    1: {
        "run": 47,
        "flue_flow": 2,
        "heat_efficiency": 48,
        "gt_prefix": "GT1",
    },
    2: {
        "run": 55,
        "flue_flow": 20,
        "heat_efficiency": 56,
        "gt_prefix": "GT2",
    },
}

GT_FEATURES = [
    "燃机热效率",
    "燃机实际功",
    "燃机计算功",
    "燃烧室温度",
    "压气机等熵效率",
    "压气机㶲效率",
    "透平等熵效率",
    "透平㶲效率",
    "压气机耗功",
    "透平输出功",
    "燃料流量",
    "空气流量",
    "燃料能量",
]

ST_FEATURES = [
    "ST.汽轮机功率",
    "ST.汽轮机计算功率",
    "ST.汽机总体等熵效率",
    "ST.高压缸等熵效率",
    "ST.中压缸等熵效率",
    "ST.高压缸出力",
    "ST.中压缸出力",
    "ST.低压缸等熵效率",
    "ST.低压缸出力",
    "ST.低压缸计算出力",
    "PLANT.供热",
    "PLANT.燃机功率",
    "PLANT.联合循环发电功率",
]


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1"])


def safe_column(df: pd.DataFrame, name: str) -> pd.Series | None:
    if name in df.columns:
        return df[name]
    return None


def build_unit_data(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    cols = UNIT_COLS[unit]
    gt_prefix = cols["gt_prefix"]

    data = pd.DataFrame(
        {
            "烟气流量": df.iloc[:, cols["flue_flow"]],
            "换热效率": df.iloc[:, cols["heat_efficiency"]],
        }
    )

    for feature in GT_FEATURES:
        col = safe_column(df, f"{gt_prefix}.{feature}")
        if col is not None:
            data[f"{gt_prefix}.{feature}"] = col

    for feature in ST_FEATURES:
        col = safe_column(df, feature)
        if col is not None:
            data[feature] = col

    mask = (
        is_true(df.iloc[:, cols["run"]])
        & data[["烟气流量", "换热效率"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["烟气流量"].between(100, 1000)
        & data["换热效率"].between(0.5, 1.05)
    )
    return data.loc[mask].copy()


def keep_high_density_points(data: pd.DataFrame) -> pd.DataFrame:
    xy = data[["烟气流量", "换热效率"]].to_numpy()
    xy_scaled = StandardScaler().fit_transform(xy)
    n_neighbors = min(K_NEIGHBORS, len(data))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(xy_scaled)
    distances, _ = nbrs.kneighbors(xy_scaled)
    density_score = distances[:, -1]
    keep_count = max(N_CLUSTERS, int(len(data) * KEEP_DENSITY_RATIO))
    keep_index = np.argsort(density_score)[:keep_count]
    kept = data.iloc[keep_index].copy()
    kept["密度得分"] = density_score[keep_index]
    return kept


def add_clusters(data: pd.DataFrame) -> pd.DataFrame:
    xy = data[["烟气流量", "换热效率"]].to_numpy()
    scaler = StandardScaler()
    xy_scaled = scaler.fit_transform(xy)
    labels = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=0).fit_predict(xy_scaled)
    data = data.copy()
    data["聚类"] = labels
    return data


def eta_squared(feature: pd.Series, labels: pd.Series) -> float:
    values = pd.to_numeric(feature, errors="coerce")
    mask = values.notna() & labels.notna()
    values = values[mask]
    labels = labels[mask]
    if values.nunique() <= 1:
        return np.nan

    overall_mean = values.mean()
    total_ss = ((values - overall_mean) ** 2).sum()
    if total_ss <= 0:
        return np.nan

    between_ss = 0.0
    for cluster_id in sorted(labels.unique()):
        group = values[labels == cluster_id]
        between_ss += len(group) * (group.mean() - overall_mean) ** 2
    return between_ss / total_ss


def print_cluster_analysis(unit: int, data: pd.DataFrame) -> None:
    print(f"\n========== HRSG{unit} ==========")
    print(f"保留高密度点数: {len(data)}")

    centers = data.groupby("聚类")[["烟气流量", "换热效率"]].agg(["count", "mean", "min", "max"])
    print("\n聚类中心特征:")
    print(centers.round(4).to_string())

    feature_cols = [
        col for col in data.columns
        if col not in ["烟气流量", "换热效率", "密度得分", "聚类"]
    ]
    scores = []
    for col in feature_cols:
        score = eta_squared(data[col], data["聚类"])
        if not pd.isna(score):
            scores.append((col, score))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)

    print("\n区分三类最明显的燃机/汽机参数:")
    for col, score in scores[:12]:
        means = data.groupby("聚类")[col].mean().round(4)
        mean_text = ", ".join(f"{idx}: {value}" for idx, value in means.items())
        print(f"{col}: eta2={score:.3f}; 各类均值 [{mean_text}]")


def plot_clusters(results: dict[int, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.2), sharex=True, sharey=True)

    for ax, unit in zip(axes, (1, 2)):
        data = results[unit]
        for cluster_id in sorted(data["聚类"].unique()):
            part = data[data["聚类"] == cluster_id]
            ax.scatter(
                part["烟气流量"],
                part["换热效率"],
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                label=f"类别{cluster_id}",
                edgecolors="none",
            )
        ax.set_title(f"HRSG{unit} 高密度点三中心聚类")
        ax.set_ylabel("换热效率")
        ax.set_xlim(100, 1000)
        ax.set_ylim(0.7, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True, markerscale=2)

    axes[-1].set_xlabel("烟气流量 / (kg/s)")
    fig.suptitle("换热效率-烟气流量高密度区域聚类分析")
    fig.tight_layout()
    plt.show()


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    results = {}
    for unit in (1, 2):
        data = build_unit_data(df, unit)
        dense_data = keep_high_density_points(data)
        clustered = add_clusters(dense_data)
        results[unit] = clustered
        print_cluster_analysis(unit, clustered)
    plot_clusters(results)


if __name__ == "__main__":
    main()
