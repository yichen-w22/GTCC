"""PPT — 烟气侧汽机发电累计能量聚类工况诊断.

基于 thesis/scripts/ch4_fluegas_energy_distribution.py 的计算逻辑。
本脚本诊断橙色点（余热锅炉出口尾气损失 + 汽机发电），不保存图片。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "Energy_Utilization_Analysis"
RAW_DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

POINT_SIZE = 5
POINT_ALPHA = 0.55
CP_FLUE_GAS = 1120.0
ST_POWER_COL = 37
HEATING_COL = 69
HEATING_THRESHOLD_MW = 1.0
N_CLUSTERS = 6

RAW_COLS = {
    "ambient_2": 10,
    "inlet_temp_2": 47,
    "outlet_temp_2": 50,
    "ambient_1": 61,
    "inlet_temp_1": 98,
    "outlet_temp_1": 101,
    "flue_flow_2": 132,
    "flue_flow_1": 133,
}

METRIC_COLS = {
    "gt1_run": 1,
    "gt1_power": 4,
    "gt2_run": 19,
    "gt2_power": 22,
}


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1"])


def sensible_heat_power(m_dot, temperature, ref_temperature):
    return m_dot * CP_FLUE_GAS * (temperature - ref_temperature)


def to_celsius(temp: pd.Series) -> pd.Series:
    if temp.median() > 200:
        return temp - 273.15
    return temp


def simple_kmeans(features: pd.DataFrame, n_clusters: int, max_iter: int = 100) -> np.ndarray:
    x = features.to_numpy(dtype=float)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    z = (x - x_mean) / x_std

    order = np.argsort(z[:, 0])
    centers = z[order[np.linspace(0, len(z) - 1, n_clusters).astype(int)]]
    labels = np.zeros(len(z), dtype=int)
    for _ in range(max_iter):
        distances = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(n_clusters):
            members = z[labels == cluster]
            if len(members) > 0:
                centers[cluster] = members.mean(axis=0)
    return labels


def build_data(raw: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    sel = raw.iloc[metrics["idx"].astype(int)].reset_index(drop=True)
    amb1 = sel.iloc[:, RAW_COLS["ambient_1"]]
    amb2 = sel.iloc[:, RAW_COLS["ambient_2"]]
    amb = sel.iloc[:, [RAW_COLS["ambient_1"], RAW_COLS["ambient_2"]]].mean(axis=1)

    flue_flow_1 = sel.iloc[:, RAW_COLS["flue_flow_1"]]
    flue_flow_2 = sel.iloc[:, RAW_COLS["flue_flow_2"]]
    outlet_temp_1 = sel.iloc[:, RAW_COLS["outlet_temp_1"]]
    outlet_temp_2 = sel.iloc[:, RAW_COLS["outlet_temp_2"]]
    inlet_temp_1 = sel.iloc[:, RAW_COLS["inlet_temp_1"]]
    inlet_temp_2 = sel.iloc[:, RAW_COLS["inlet_temp_2"]]

    inlet_energy = (
        sensible_heat_power(flue_flow_1, inlet_temp_1, amb)
        + sensible_heat_power(flue_flow_2, inlet_temp_2, amb)
    )
    stack_loss_1 = sensible_heat_power(flue_flow_1, outlet_temp_1, amb)
    stack_loss_2 = sensible_heat_power(flue_flow_2, outlet_temp_2, amb)
    stack_loss = stack_loss_1 + stack_loss_2
    generation = metrics.iloc[:, ST_POWER_COL]
    condenser = inlet_energy - stack_loss - generation

    gt1_run = is_true(metrics.iloc[:, METRIC_COLS["gt1_run"]])
    gt2_run = is_true(metrics.iloc[:, METRIC_COLS["gt2_run"]])
    gt_mode = np.select(
        [gt1_run & gt2_run, gt1_run & ~gt2_run, ~gt1_run & gt2_run],
        ["双燃机运行", "仅GT1运行", "仅GT2运行"],
        default="燃机运行状态异常",
    )

    data = pd.DataFrame(
        {
            "原始序号": metrics["idx"].astype(int),
            "汽机出力_MW": generation / 1e6,
            "尾气损失_MW": stack_loss / 1e6,
            "汽机发电累计_MW": (stack_loss + generation) / 1e6,
            "HRSG1尾气损失_MW": stack_loss_1 / 1e6,
            "HRSG2尾气损失_MW": stack_loss_2 / 1e6,
            "凝汽器放热_热网供热_MW": condenser / 1e6,
            "入口烟气能量_MW": inlet_energy / 1e6,
            "供热量_MW": metrics.iloc[:, HEATING_COL].fillna(0).clip(lower=0) / 1e6,
            "GT1功率_MW": metrics.iloc[:, METRIC_COLS["gt1_power"]] / 1e6,
            "GT2功率_MW": metrics.iloc[:, METRIC_COLS["gt2_power"]] / 1e6,
            "燃机运行方式": gt_mode,
            "烟气流量1": flue_flow_1,
            "烟气流量2": flue_flow_2,
            "烟气总流量": flue_flow_1.fillna(0) + flue_flow_2.fillna(0),
            "排烟温度1_C": to_celsius(outlet_temp_1),
            "排烟温度2_C": to_celsius(outlet_temp_2),
            "排烟平均温度_C": to_celsius(pd.concat([outlet_temp_1, outlet_temp_2], axis=1).mean(axis=1)),
            "进口烟温1_C": to_celsius(inlet_temp_1),
            "进口烟温2_C": to_celsius(inlet_temp_2),
            "环境温度_C": to_celsius(pd.concat([amb1, amb2], axis=1).mean(axis=1)),
        }
    )
    data["供热状态"] = np.where(data["供热量_MW"] > HEATING_THRESHOLD_MW, "供热工况", "非供热工况")
    data["季节"] = np.where(data["环境温度_C"] >= 20, "夏季/高温", "冬季/低温")

    mask = (
        data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["汽机出力_MW"].between(40, 300)
        & (data["入口烟气能量_MW"] > 50)
        & (data["尾气损失_MW"] > 0)
        & (data["凝汽器放热_热网供热_MW"] > 0)
    )
    return data.loc[mask].reset_index(drop=True)


def dominant_text(series: pd.Series) -> str:
    counts = series.value_counts(normalize=True)
    if counts.empty:
        return ""
    return f"{counts.index[0]} ({counts.iloc[0] * 100:.1f}%)"


def summarize_clusters(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_cols = [
        "汽机出力_MW",
        "尾气损失_MW",
        "汽机发电累计_MW",
        "趋势线残差_MW",
        "供热量_MW",
        "烟气总流量",
        "排烟平均温度_C",
        "环境温度_C",
        "GT1功率_MW",
        "GT2功率_MW",
    ]
    for cluster, part in data.groupby("聚类", sort=True):
        row = {
            "聚类": cluster,
            "样本数": len(part),
            "主要燃机运行方式": dominant_text(part["燃机运行方式"]),
            "主要供热状态": dominant_text(part["供热状态"]),
            "主要季节": dominant_text(part["季节"]),
        }
        for col in numeric_cols:
            row[f"{col}_中位数"] = part[col].median()
            row[f"{col}_范围"] = f"{part[col].min():.1f}-{part[col].max():.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def print_cluster_details(data: pd.DataFrame) -> None:
    summary = summarize_clusters(data)
    display_cols = [
        "聚类",
        "样本数",
        "主要燃机运行方式",
        "主要供热状态",
        "主要季节",
        "汽机出力_MW_范围",
        "汽机发电累计_MW_范围",
        "趋势线残差_MW_范围",
        "供热量_MW_中位数",
        "烟气总流量_中位数",
        "排烟平均温度_C_中位数",
        "环境温度_C_中位数",
    ]
    print("\n橙色点聚类对应工况")
    print(summary[display_cols].to_string(index=False))

    print("\n每个聚类内的燃机运行方式占比")
    print(pd.crosstab(data["聚类"], data["燃机运行方式"], normalize="index").mul(100).round(1).to_string())

    print("\n每个聚类内的供热状态占比")
    print(pd.crosstab(data["聚类"], data["供热状态"], normalize="index").mul(100).round(1).to_string())

    print("\n每个聚类内的季节占比")
    print(pd.crosstab(data["聚类"], data["季节"], normalize="index").mul(100).round(1).to_string())


def fit_trend_line(data: pd.DataFrame) -> tuple[float, float]:
    slope, intercept = np.polyfit(data["汽机出力_MW"], data["汽机发电累计_MW"], 1)
    data["趋势线_MW"] = slope * data["汽机出力_MW"] + intercept
    data["趋势线残差_MW"] = data["汽机发电累计_MW"] - data["趋势线_MW"]
    return slope, intercept


def draw_cluster_plot(data: pd.DataFrame, slope: float, intercept: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    sc = ax.scatter(
        data["汽机出力_MW"],
        data["汽机发电累计_MW"],
        c=data["聚类"],
        cmap="tab10",
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="none",
    )
    x_line = np.linspace(data["汽机出力_MW"].min(), data["汽机出力_MW"].max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, label="趋势线")
    ax.set_xlabel("汽机出力 / MW")
    ax.set_ylabel("尾气损失 + 汽机发电 / MW")
    ax.set_title("橙色点趋势线残差聚类诊断")
    ax.set_xlim(40, 300)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("聚类编号")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def draw_condition_plots(data: pd.DataFrame) -> None:
    plots = [
        ("供热量_MW", "供热量 / MW"),
        ("环境温度_C", "环境温度 / ℃"),
        ("烟气总流量", "烟气总流量"),
        ("排烟平均温度_C", "排烟平均温度 / ℃"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
    for ax, (col, label) in zip(axes.ravel(), plots):
        sc = ax.scatter(
            data["汽机出力_MW"],
            data["汽机发电累计_MW"],
            c=data[col],
            cmap="viridis",
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="none",
        )
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax)
    for ax in axes[-1]:
        ax.set_xlabel("汽机出力 / MW")
    for ax in axes[:, 0]:
        ax.set_ylabel("尾气损失 + 汽机发电 / MW")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main() -> None:
    raw = pd.read_csv(RAW_DATA_PATH)
    metrics = pd.read_csv(PLANT_METRICS_PATH)
    data = build_data(raw, metrics)
    slope, intercept = fit_trend_line(data)
    data["聚类"] = simple_kmeans(data[["汽机出力_MW", "趋势线残差_MW"]], N_CLUSTERS)

    # 按汽机出力中位数、趋势线残差中位数重新编号，便于从左到右、从下到上阅读。
    order = (
        data.groupby("聚类")[["汽机出力_MW", "趋势线残差_MW"]]
        .median()
        .sort_values(["汽机出力_MW", "趋势线残差_MW"])
        .index
    )
    remap = {old: new + 1 for new, old in enumerate(order)}
    data["聚类"] = data["聚类"].map(remap)

    print(f"橙色点趋势线：y = {slope:.4f} * 汽机出力 + {intercept:.4f}")
    print("聚类指标：汽机出力_MW + 趋势线残差_MW")
    print_cluster_details(data)
    draw_cluster_plot(data, slope, intercept)
    draw_condition_plots(data)


if __name__ == "__main__":
    main()
