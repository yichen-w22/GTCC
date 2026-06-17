"""Chapter 3 — 燃气轮机部件性能 (3燃机1压气机.png, 3燃机1透平.png, 4燃机2压气机.png, 4燃机2透平.png)"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1].parent / "Energy_Utilization_Analysis"
RAW_DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "chapter3"

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

POINT_SIZE = 4
POINT_ALPHA = 0.40
POWER_LOWER_MW = 100.0
POWER_UPPER_MW = 300.0

COLS = {
    1: {"run": 1, "actual_power": 4, "compressor_eff": 7, "turbine_eff": 9},
    2: {"run": 19, "actual_power": 22, "compressor_eff": 25, "turbine_eff": 27},
}


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1"])


def plot_single(x, y, t, vmin, vmax, xlabel, ylabel, fname):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sc = ax.scatter(
        x, y, c=t, cmap="coolwarm", vmin=vmin, vmax=vmax,
        s=POINT_SIZE, alpha=POINT_ALPHA, edgecolors="none",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("环境温度 / ℃")
    fig.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"已保存 {OUTPUT_DIR / fname}")


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    raw_df = pd.read_csv(RAW_DATA_PATH)

    idx = df["idx"].astype(int)
    temp1 = raw_df.loc[idx, "环境温度_1"].reset_index(drop=True)
    temp2 = raw_df.loc[idx, "环境温度_2"].reset_index(drop=True)
    if temp1.median() > 200:
        temp1 = temp1 - 273.15
    if temp2.median() > 200:
        temp2 = temp2 - 273.15

    cols1, cols2 = COLS[1], COLS[2]
    mask1 = (
        is_true(df.iloc[:, cols1["run"]])
        & df.iloc[:, cols1["actual_power"]].between(POWER_LOWER_MW * 1e6, POWER_UPPER_MW * 1e6)
        & df.iloc[:, cols1["compressor_eff"]].between(0.5, 1.0)
        & df.iloc[:, cols1["turbine_eff"]].between(0.5, 1.0)
    )
    mask2 = (
        is_true(df.iloc[:, cols2["run"]])
        & df.iloc[:, cols2["actual_power"]].between(POWER_LOWER_MW * 1e6, POWER_UPPER_MW * 1e6)
        & df.iloc[:, cols2["compressor_eff"]].between(0.5, 1.0)
        & df.iloc[:, cols2["turbine_eff"]].between(0.5, 1.0)
    )
    vmin = min(temp1[mask1].quantile(0.01), temp2[mask2].quantile(0.01))
    vmax = max(temp1[mask1].quantile(0.99), temp2[mask2].quantile(0.99))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for unit, cols, mask, temp in [(1, cols1, mask1, temp1), (2, cols2, mask2, temp2)]:
        x = df.iloc[mask.values, cols["actual_power"]] / 1e6
        comp_eff = df.iloc[mask.values, cols["compressor_eff"]]
        turb_eff = df.iloc[mask.values, cols["turbine_eff"]]
        t = temp[mask].reset_index(drop=True)
        prefix = "3" if unit == 1 else "4"
        plot_single(x, comp_eff, t, vmin, vmax, "燃机实际出力 / MW", "压气机等熵效率", f"{prefix}燃机{unit}压气机.png")
        plot_single(x, turb_eff, t, vmin, vmax, "燃机实际出力 / MW", "透平等熵效率", f"{prefix}燃机{unit}透平.png")


if __name__ == "__main__":
    main()
