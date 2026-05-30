from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

HEATING_THRESHOLD = 1.0e6

METRIC_COLS = {
    "联合循环发电效率": 64,
    "联合循环热电效率": 65,
    "整机出力": 63,
    "燃机总出力": 66,
    "汽机出力": 67,
    "供热量": 69,
}

RAW_COLS = {
    "环境温度_2": 10,
    "压气机入口压力_2": 1,
    "大气相对湿度_2": 8,
    "高压给水压力_2": 11,
    "中压给水压力_2": 20,
    "低压给水压力_2": 38,
    "高压主蒸汽温度_2": 17,
    "中压主蒸汽温度_2": 26,
    "低压主蒸汽温度_2": 43,
    "高压主蒸汽流量_2": 19,
    "中压主蒸汽流量_2": 24,
    "低压主蒸汽流量_2": 44,
    "透平排气温度_2": 9,
    "空气流量_2": 137,
    "环境温度_1": 61,
    "压气机入口压力_1": 52,
    "大气相对湿度_1": 59,
    "高压给水压力_1": 62,
    "中压给水压力_1": 71,
    "低压给水压力_1": 89,
    "高压主蒸汽温度_1": 68,
    "中压主蒸汽温度_1": 77,
    "低压主蒸汽温度_1": 94,
    "高压主蒸汽流量_1": 70,
    "中压主蒸汽流量_1": 75,
    "低压主蒸汽流量_1": 95,
    "透平排气温度_1": 60,
    "空气流量_1": 141,
}


def average_pair(raw: pd.DataFrame, name_1: str, name_2: str) -> pd.Series:
    return raw.iloc[:, [RAW_COLS[name_1], RAW_COLS[name_2]]].mean(axis=1)


def sum_pair(raw: pd.DataFrame, name_1: str, name_2: str) -> pd.Series:
    return raw.iloc[:, RAW_COLS[name_1]].fillna(0) + raw.iloc[:, RAW_COLS[name_2]].fillna(0)


def build_correlation_data(raw: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    selected_raw = raw.iloc[metrics["idx"].astype(int)].reset_index(drop=True)

    data = pd.DataFrame(
        {
            "联合循环发电效率": metrics.iloc[:, METRIC_COLS["联合循环发电效率"]] * 100,
            "联合循环热电效率": metrics.iloc[:, METRIC_COLS["联合循环热电效率"]] * 100,
            "整机出力": metrics.iloc[:, METRIC_COLS["整机出力"]] / 1e6,
            "燃机总出力": metrics.iloc[:, METRIC_COLS["燃机总出力"]] / 1e6,
            "汽机出力": metrics.iloc[:, METRIC_COLS["汽机出力"]] / 1e6,
            "供热量": metrics.iloc[:, METRIC_COLS["供热量"]].fillna(0).clip(lower=0) / 1e6,
            "环境温度": average_pair(selected_raw, "环境温度_1", "环境温度_2"),
            "环境压力": average_pair(selected_raw, "压气机入口压力_1", "压气机入口压力_2"),
            "环境湿度": average_pair(selected_raw, "大气相对湿度_1", "大气相对湿度_2"),
            "高压给水压力": average_pair(selected_raw, "高压给水压力_1", "高压给水压力_2"),
            "中压给水压力": average_pair(selected_raw, "中压给水压力_1", "中压给水压力_2"),
            "低压给水压力": average_pair(selected_raw, "低压给水压力_1", "低压给水压力_2"),
            "高压主蒸汽温度": average_pair(selected_raw, "高压主蒸汽温度_1", "高压主蒸汽温度_2"),
            "中压主蒸汽温度": average_pair(selected_raw, "中压主蒸汽温度_1", "中压主蒸汽温度_2"),
            "低压主蒸汽温度": average_pair(selected_raw, "低压主蒸汽温度_1", "低压主蒸汽温度_2"),
            "高压主蒸汽流量": sum_pair(selected_raw, "高压主蒸汽流量_1", "高压主蒸汽流量_2"),
            "中压主蒸汽流量": sum_pair(selected_raw, "中压主蒸汽流量_1", "中压主蒸汽流量_2"),
            "低压主蒸汽流量": sum_pair(selected_raw, "低压主蒸汽流量_1", "低压主蒸汽流量_2"),
            "空气流量": sum_pair(selected_raw, "空气流量_1", "空气流量_2"),
            "透平排气温度": average_pair(selected_raw, "透平排气温度_1", "透平排气温度_2"),
        }
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def filter_condition_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_mask = data.notna().all(axis=1) & data["整机出力"].between(0, 800)
    non_heating = data.loc[
        base_mask
        & (data["供热量"] <= HEATING_THRESHOLD / 1e6)
        & data["联合循环发电效率"].between(0, 100)
    ].copy()
    heating = data.loc[
        base_mask
        & (data["供热量"] > HEATING_THRESHOLD / 1e6)
        & data["联合循环热电效率"].between(0, 120)
    ].copy()

    print(f"非供热工况相关性样本数 {len(non_heating)}")
    print(f"供热工况相关性样本数 {len(heating)}")
    return non_heating, heating


def draw_corr_matrix(ax, corr: pd.DataFrame, title: str) -> None:
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            color = "white" if abs(value) > 0.65 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=6)

    return im


def plot_correlation_matrices(data: pd.DataFrame) -> None:
    non_heating, heating = filter_condition_data(data)

    corr_non_heating = non_heating.corr(method="pearson")
    corr_heating = heating.corr(method="pearson")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    im = draw_corr_matrix(axes[0], corr_non_heating, "非供热工况")
    draw_corr_matrix(axes[1], corr_heating, "供热工况")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Pearson相关系数")

    plt.show()


def main() -> None:
    raw = pd.read_csv(RAW_DATA_PATH)
    metrics = pd.read_csv(PLANT_METRICS_PATH)
    data = build_correlation_data(raw, metrics)
    plot_correlation_matrices(data)


if __name__ == "__main__":
    main()
