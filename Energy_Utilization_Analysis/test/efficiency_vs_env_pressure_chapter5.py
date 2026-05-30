from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"

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
HEATING_COL = 69

HP_STEAM_FLOW_2_COL = 19
HP_STEAM_FLOW_1_COL = 70


def build_plot_data(raw: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_raw = raw.iloc[metrics["idx"].astype(int)].reset_index(drop=True)
    hp_steam_flow = (
        selected_raw.iloc[:, HP_STEAM_FLOW_1_COL].fillna(0)
        + selected_raw.iloc[:, HP_STEAM_FLOW_2_COL].fillna(0)
    )

    data = pd.DataFrame(
        {
            "hp_steam_flow": hp_steam_flow,
            "power_efficiency_pct": metrics.iloc[:, POWER_EFFICIENCY_COL] * 100,
            "heat_power_efficiency_pct": metrics.iloc[:, HEAT_POWER_EFFICIENCY_COL] * 100,
            "plant_power_mw": metrics.iloc[:, PLANT_POWER_COL] / 1e6,
            "heating_mw": metrics.iloc[:, HEATING_COL].fillna(0).clip(lower=0) / 1e6,
        }
    )

    base_mask = (
        data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["plant_power_mw"].between(0, 800)
        & (data["hp_steam_flow"] > 0)
    )
    non_heating = data.loc[
        base_mask
        & (data["heating_mw"] <= HEATING_THRESHOLD / 1e6)
        & data["power_efficiency_pct"].between(0, 100),
        ["hp_steam_flow", "power_efficiency_pct"],
    ].copy()
    heating = data.loc[
        base_mask
        & (data["heating_mw"] > HEATING_THRESHOLD / 1e6)
        & data["heat_power_efficiency_pct"].between(0, 120),
        ["hp_steam_flow", "heat_power_efficiency_pct"],
    ].copy()

    print(f"非供热工况绘图点数 {len(non_heating)}")
    print(f"供热工况绘图点数 {len(heating)}")
    return non_heating, heating


def plot_efficiency_vs_hp_steam_flow(raw: pd.DataFrame, metrics: pd.DataFrame) -> None:
    non_heating, heating = build_plot_data(raw, metrics)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        non_heating["hp_steam_flow"],
        non_heating["power_efficiency_pct"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label="非供热工况",
        edgecolors="none",
    )
    ax.scatter(
        heating["hp_steam_flow"],
        heating["heat_power_efficiency_pct"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label="供热工况",
        edgecolors="none",
    )
    ax.set_xlabel("高压主蒸汽流量 / kg/s")
    ax.set_ylabel("联合循环效率 / %")
    ax.set_ylim(20, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, markerscale=3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    raw = pd.read_csv(RAW_DATA_PATH)
    metrics = pd.read_csv(PLANT_METRICS_PATH)
    plot_efficiency_vs_hp_steam_flow(raw, metrics)


if __name__ == "__main__":
    main()
