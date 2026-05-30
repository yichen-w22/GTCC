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

ST_POWER_COL = 37
GT1_FUEL_FLOW_COL = 13
GT2_FUEL_FLOW_COL = 31


def build_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "st_power_mw": df.iloc[:, ST_POWER_COL] / 1e6,
            "fuel_flow": df.iloc[:, GT1_FUEL_FLOW_COL].fillna(0)
            + df.iloc[:, GT2_FUEL_FLOW_COL].fillna(0),
        }
    )

    mask = (
        data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["st_power_mw"].between(0, 300)
        & (data["fuel_flow"] > 0)
    )

    data = data.loc[mask].sort_values("st_power_mw").reset_index(drop=True)
    print(f"绘图点数 {len(data)}")
    return data


def plot_fuel_flow_vs_st_power(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    data = build_plot_data(df)
    ax.scatter(
        data["st_power_mw"],
        data["fuel_flow"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="none",
    )

    ax.set_xlabel("汽机出力 / MW")
    ax.set_ylabel("总燃料流量 / kg/s")
    ax.set_xlim(0, 300)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    plot_fuel_flow_vs_st_power(df)


if __name__ == "__main__":
    main()
