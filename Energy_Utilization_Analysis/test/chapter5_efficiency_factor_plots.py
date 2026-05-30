from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
RAW_DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"
OUTPUT_DIR = REPO_ROOT / "thesis" / "figures" / "chapter5"

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

POINT_SIZE = 4
POINT_ALPHA = 0.30
HEATING_THRESHOLD = 1.0e6

POWER_EFFICIENCY_COL = 64
HEAT_POWER_EFFICIENCY_COL = 65
PLANT_POWER_COL = 63
GT_POWER_COL = 66
ST_POWER_COL = 67
HEATING_COL = 69

RAW_COLS = {
    "amb_temp_2": 10,
    "amb_pressure_2": 1,
    "amb_humidity_2": 8,
    "hp_fw_pressure_2": 11,
    "ip_fw_pressure_2": 20,
    "lp_fw_pressure_2": 38,
    "hp_steam_flow_2": 19,
    "ip_steam_flow_2": 24,
    "exhaust_temp_2": 9,
    "air_flow_2": 137,
    "amb_temp_1": 61,
    "amb_pressure_1": 52,
    "amb_humidity_1": 59,
    "hp_fw_pressure_1": 62,
    "ip_fw_pressure_1": 71,
    "lp_fw_pressure_1": 89,
    "hp_steam_flow_1": 70,
    "ip_steam_flow_1": 75,
    "exhaust_temp_1": 60,
    "air_flow_1": 141,
}

FACTOR_ORDER = [
    "汽机出力",
    "环境压力",
    "环境温度",
    "低压给水压力",
    "整机出力",
    "高压主蒸汽流量",
    "高压给水压力",
    "中压主蒸汽流量",
    "中压给水压力",
    "透平排气温度",
    "供热量",
    "环境湿度",
    "空气流量",
    "燃机总出力",
]

FACTOR_LABELS = {
    "汽机出力": "汽机出力 / MW",
    "环境压力": "环境压力 / kPa",
    "环境温度": "环境温度 / ℃",
    "低压给水压力": "低压给水压力 / MPa",
    "整机出力": "整机出力 / MW",
    "高压主蒸汽流量": "高压主蒸汽流量 / kg/s",
    "高压给水压力": "高压给水压力 / MPa",
    "中压主蒸汽流量": "中压主蒸汽流量 / kg/s",
    "中压给水压力": "中压给水压力 / MPa",
    "透平排气温度": "透平排气温度 / ℃",
    "供热量": "供热量 / MW",
    "环境湿度": "环境湿度 / %",
    "空气流量": "空气流量 / kg/s",
    "燃机总出力": "燃机总出力 / MW",
}


def average_pair(raw: pd.DataFrame, name_1: str, name_2: str) -> pd.Series:
    return raw.iloc[:, [RAW_COLS[name_1], RAW_COLS[name_2]]].mean(axis=1)


def sum_pair(raw: pd.DataFrame, name_1: str, name_2: str) -> pd.Series:
    return raw.iloc[:, RAW_COLS[name_1]].fillna(0) + raw.iloc[:, RAW_COLS[name_2]].fillna(0)


def build_factor_data(raw: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    selected_raw = raw.iloc[metrics["idx"].astype(int)].reset_index(drop=True)

    data = pd.DataFrame(
        {
            "联合循环发电效率": metrics.iloc[:, POWER_EFFICIENCY_COL] * 100,
            "联合循环热电效率": metrics.iloc[:, HEAT_POWER_EFFICIENCY_COL] * 100,
            "整机出力": metrics.iloc[:, PLANT_POWER_COL] / 1e6,
            "燃机总出力": metrics.iloc[:, GT_POWER_COL] / 1e6,
            "汽机出力": metrics.iloc[:, ST_POWER_COL] / 1e6,
            "供热量": metrics.iloc[:, HEATING_COL].fillna(0).clip(lower=0) / 1e6,
            "环境温度": average_pair(selected_raw, "amb_temp_1", "amb_temp_2") - 273.15,
            "环境压力": average_pair(selected_raw, "amb_pressure_1", "amb_pressure_2") / 1000,
            "环境湿度": average_pair(selected_raw, "amb_humidity_1", "amb_humidity_2"),
            "高压给水压力": average_pair(selected_raw, "hp_fw_pressure_1", "hp_fw_pressure_2") / 1e6,
            "中压给水压力": average_pair(selected_raw, "ip_fw_pressure_1", "ip_fw_pressure_2") / 1e6,
            "低压给水压力": average_pair(selected_raw, "lp_fw_pressure_1", "lp_fw_pressure_2") / 1e6,
            "高压主蒸汽流量": sum_pair(selected_raw, "hp_steam_flow_1", "hp_steam_flow_2"),
            "中压主蒸汽流量": sum_pair(selected_raw, "ip_steam_flow_1", "ip_steam_flow_2"),
            "空气流量": sum_pair(selected_raw, "air_flow_1", "air_flow_2"),
            "透平排气温度": average_pair(selected_raw, "exhaust_temp_1", "exhaust_temp_2") - 273.15,
        }
    )

    return data.replace([np.inf, -np.inf], np.nan)


def split_conditions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    print(f"非供热工况样本数 {len(non_heating)}")
    print(f"供热工况样本数 {len(heating)}")
    return non_heating, heating


def draw_factor_plot(factor: str, non_heating: pd.DataFrame, heating: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        non_heating[factor],
        non_heating["联合循环发电效率"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label="非供热工况",
        edgecolors="none",
    )
    ax.scatter(
        heating[factor],
        heating["联合循环热电效率"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label="供热工况",
        edgecolors="none",
    )

    ax.set_xlabel(FACTOR_LABELS[factor])
    ax.set_ylabel("联合循环效率 / %")
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, markerscale=3)

    fig.tight_layout()
    output_path = OUTPUT_DIR / f"{factor}_效率关系.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"已保存 {output_path}")


def build_correlation_summary(non_heating: pd.DataFrame, heating: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in FACTOR_ORDER:
        non_corr = non_heating[[factor, "联合循环发电效率"]].corr().iloc[0, 1]
        heating_corr = heating[[factor, "联合循环热电效率"]].corr().iloc[0, 1]
        rows.append(
            {
                "影响因素": factor,
                "非供热工况-联合循环发电效率相关系数": non_corr,
                "供热工况-联合循环热电效率相关系数": heating_corr,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_DATA_PATH)
    metrics = pd.read_csv(PLANT_METRICS_PATH)
    data = build_factor_data(raw, metrics)
    non_heating, heating = split_conditions(data)

    for factor in FACTOR_ORDER:
        draw_factor_plot(factor, non_heating, heating)

    summary = build_correlation_summary(non_heating, heating)
    summary_path = OUTPUT_DIR / "效率影响因素相关系数汇总.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"已保存 {summary_path}")


if __name__ == "__main__":
    main()
