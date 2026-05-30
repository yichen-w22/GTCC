from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

PLANT_POWER_COL = 63
POWER_EFFICIENCY_COL = 64
HEAT_POWER_EFFICIENCY_COL = 65
HEATING_COL = 69


def build_plot_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.DataFrame(
        {
            "plant_power_mw": df.iloc[:, PLANT_POWER_COL] / 1e6,
            "power_efficiency_pct": df.iloc[:, POWER_EFFICIENCY_COL] * 100,
            "heat_power_efficiency_pct": df.iloc[:, HEAT_POWER_EFFICIENCY_COL] * 100,
            "heating_mw": df.iloc[:, HEATING_COL].fillna(0).clip(lower=0) / 1e6,
        }
    )

    base_mask = (
        data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["plant_power_mw"].between(0, 800)
    )
    non_heating = data.loc[
        base_mask
        & (data["heating_mw"] <= HEATING_THRESHOLD / 1e6)
        & data["power_efficiency_pct"].between(0, 100),
        ["plant_power_mw", "power_efficiency_pct"],
    ].copy()
    heating = data.loc[
        base_mask
        & (data["heating_mw"] > HEATING_THRESHOLD / 1e6)
        & data["heat_power_efficiency_pct"].between(0, 120),
        ["plant_power_mw", "heat_power_efficiency_pct"],
    ].copy()

    non_heating = non_heating.sort_values("plant_power_mw").reset_index(drop=True)
    heating = heating.sort_values("plant_power_mw").reset_index(drop=True)
    print(f"非供热工况绘图点数 {len(non_heating)}")
    print(f"供热工况绘图点数 {len(heating)}")
    return non_heating, heating


def plot_plant_efficiency_vs_power(df: pd.DataFrame) -> None:
    non_heating, heating = build_plot_data(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    axes[0].scatter(
        non_heating["plant_power_mw"],
        non_heating["power_efficiency_pct"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="none",
    )
    axes[0].set_title("非供热工况")
    axes[0].set_xlabel("联合循环发电功率 / MW")
    axes[0].set_ylabel("联合循环发电效率 / %")
    axes[0].set_xlim(0, 800)
    axes[0].set_ylim(30, 100)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        heating["plant_power_mw"],
        heating["heat_power_efficiency_pct"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="none",
    )
    axes[1].set_title("供热工况")
    axes[1].set_xlabel("联合循环发电功率 / MW")
    axes[1].set_ylabel("联合循环热电效率 / %")
    axes[1].set_xlim(0, 800)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    plot_plant_efficiency_vs_power(df)


if __name__ == "__main__":
    main()
