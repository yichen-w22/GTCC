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

POWER_LOWER_MW = 100.0
POWER_UPPER_MW = 300.0
POINT_SIZE = 4
POINT_ALPHA = 0.32

COLS = {
    1: {
        "run": 1,
        "actual_power": 4,
        "chamber_temperature": 6,
        "compressor_efficiency": 7,
        "turbine_efficiency": 9,
        "compressor_power": 11,
        "fuel_flow": 13,
        "air_flow": 14,
        "fuel_energy": 15,
    },
    2: {
        "run": 19,
        "actual_power": 22,
        "chamber_temperature": 24,
        "compressor_efficiency": 25,
        "turbine_efficiency": 27,
        "compressor_power": 29,
        "fuel_flow": 31,
        "air_flow": 32,
        "fuel_energy": 33,
    },
}


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1"])


def build_stack_data(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    cols = COLS[unit]
    actual_power = df.iloc[:, cols["actual_power"]]
    compressor_power = df.iloc[:, cols["compressor_power"]]
    fuel_energy = df.iloc[:, cols["fuel_energy"]]
    exhaust_energy = fuel_energy - actual_power - compressor_power

    data = pd.DataFrame(
        {
            "actual_power_mw": actual_power / 1e6,
            "generation_power_mw": actual_power / 1e6,
            "compressor_power_mw": compressor_power / 1e6,
            "exhaust_energy_mw": exhaust_energy / 1e6,
            "fuel_energy_mw": fuel_energy / 1e6,
        }
    )

    finite_mask = data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    mask = (
        is_true(df.iloc[:, cols["run"]])
        & finite_mask
        & data["actual_power_mw"].between(POWER_LOWER_MW, POWER_UPPER_MW)
        & (data["generation_power_mw"] > 0)
        & (data["compressor_power_mw"] > 0)
        & (data["exhaust_energy_mw"] > 0)
        & (data["fuel_energy_mw"] > 0)
    )

    data = data.loc[mask].copy()
    print(f"GT{unit}: 绘图点数 {len(data)}")

    return data.sort_values("actual_power_mw").reset_index(drop=True)


def plot_stack(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    colors = ["#1B4F72", "#5DA5DA", "#A23E3E"]
    labels = ["发电功率", "压气机耗功", "尾气能量及其他损失"]

    for ax, unit in zip(axes, (1, 2)):
        data = build_stack_data(df, unit)
        generation_top = data["generation_power_mw"]
        compressor_top = generation_top + data["compressor_power_mw"]
        exhaust_top = compressor_top + data["exhaust_energy_mw"]

        ax.scatter(
            data["actual_power_mw"],
            generation_top,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=colors[0],
            label=labels[0],
            edgecolors="none",
        )
        ax.scatter(
            data["actual_power_mw"],
            compressor_top,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=colors[1],
            label=labels[1],
            edgecolors="none",
        )
        ax.scatter(
            data["actual_power_mw"],
            exhaust_top,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=colors[2],
            label=labels[2],
            edgecolors="none",
        )

        ax.set_title(f"GT{unit} 能量分配")
        ax.set_xlabel("燃机实际出力 / MW")
        ax.set_xlim(POWER_LOWER_MW, POWER_UPPER_MW)
        ax.set_ylim(0, 900)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("累计能量流率 / MW")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend = axes[0].legend(
        handles[::-1],
        legend_labels[::-1],
        loc="upper left",
        frameon=True,
        scatterpoints=1,
        markerscale=3,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    fig.tight_layout()
    plt.show()


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    plot_stack(df)


if __name__ == "__main__":
    main()
