"""PPT — 燃气轮机能量占比散点图，并标注性能试验热效率点.

本脚本基于 thesis/scripts/ch3_gt_energy_share.py，新建于 PPT 目录；
只使用 plt.show() 显示图形，不保存图片。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "Energy_Utilization_Analysis"
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

POWER_LOWER_MW = 100.0
POWER_UPPER_MW = 300.0
POINT_SIZE = 4
POINT_ALPHA = 0.32

COLS = {
    1: {"run": 1, "actual_power": 4, "compressor_power": 11, "fuel_energy": 15},
    2: {"run": 19, "actual_power": 22, "compressor_power": 29, "fuel_energy": 33},
}

PERFORMANCE_TEST_GT1 = pd.DataFrame(
    {
        "load": ["100%负荷", "80%负荷", "75%负荷", "65%负荷"],
        "power_mw": [277.25, 217.94, 194.53, 172.37],
        "thermal_efficiency": [37.94, 36.71, 35.47, 34.09],
    }
)


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
            "generation_share": actual_power / fuel_energy,
            "compressor_share": compressor_power / fuel_energy,
            "exhaust_share": exhaust_energy / fuel_energy,
        }
    )

    finite_mask = data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    mask = (
        is_true(df.iloc[:, cols["run"]])
        & finite_mask
        & data["actual_power_mw"].between(POWER_LOWER_MW, POWER_UPPER_MW)
        & (data["generation_share"] > 0)
        & (data["compressor_share"] > 0)
        & (data["exhaust_share"] > 0)
    )
    return data.loc[mask].sort_values("actual_power_mw").reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)

    data = build_stack_data(df, 1)
    test = PERFORMANCE_TEST_GT1.copy()
    test["thermal_efficiency"] = test["thermal_efficiency"] / 100
    generation_top = data["generation_share"]
    compressor_top = generation_top + data["compressor_share"]
    exhaust_top = compressor_top + data["exhaust_share"]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(
        data["actual_power_mw"],
        generation_top,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label="燃机出力",
        edgecolors="none",
    )
    ax.scatter(
        data["actual_power_mw"],
        compressor_top,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label="压气机耗功",
        edgecolors="none",
    )
    ax.scatter(
        data["actual_power_mw"],
        exhaust_top,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label="尾气能量及其他损失",
        edgecolors="none",
    )
    ax.scatter(
        test["power_mw"],
        test["thermal_efficiency"],
        marker="x",
        s=80,
        linewidths=2,
        color="black",
        label="性能试验",
        zorder=5,
    )

    ax.set_xlabel("燃机实际出力 / MW")
    ax.set_ylabel("累计能量占比")
    ax.set_title("GT1 能量占比与性能试验点")
    ax.set_xlim(POWER_LOWER_MW, POWER_UPPER_MW)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3)
    handles, legend_labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], legend_labels[::-1], loc="upper left", frameon=True, markerscale=1.5)
    legend.get_frame().set_alpha(0.3)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
