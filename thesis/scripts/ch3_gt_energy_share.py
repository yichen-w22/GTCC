"""Chapter 3 — 燃气轮机能量占比散点图 (2-1GT1能量占比.png, 2-1GT2能量占比.png)"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1].parent / "Energy_Utilization_Analysis"
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "chapter3"

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


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1"])


def build_stack_data(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    cols = COLS[unit]
    actual_power = df.iloc[:, cols["actual_power"]]
    compressor_power = df.iloc[:, cols["compressor_power"]]
    fuel_energy = df.iloc[:, cols["fuel_energy"]]
    exhaust_energy = fuel_energy - actual_power - compressor_power

    data = pd.DataFrame({
        "actual_power_mw": actual_power / 1e6,
        "generation_power_mw": actual_power / 1e6,
        "compressor_power_mw": compressor_power / 1e6,
        "exhaust_energy_mw": exhaust_energy / 1e6,
        "fuel_energy_mw": fuel_energy / 1e6,
    })

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
    return data.loc[mask].sort_values("actual_power_mw").reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    labels = ["压气机耗功占比", "发电功率占比", "尾气能量及其他损失占比"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for unit, fname in [(1, "2-1GT1能量占比.png"), (2, "2-1GT2能量占比.png")]:
        data = build_stack_data(df, unit)
        total = data["fuel_energy_mw"]
        comp_share = data["compressor_power_mw"] / total
        gen_share = comp_share + data["generation_power_mw"] / total
        exh_share = gen_share + data["exhaust_energy_mw"] / total

        fig, ax = plt.subplots(figsize=(5, 4.5))
        for y, label in zip([comp_share, gen_share, exh_share], labels):
            ax.scatter(data["actual_power_mw"], y, s=POINT_SIZE, alpha=POINT_ALPHA, label=label, edgecolors="none")

        ax.set_xlabel("燃机实际出力 / MW")
        ax.set_ylabel("累计能量占比")
        ax.set_xlim(POWER_LOWER_MW, POWER_UPPER_MW)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3)
        handles, legend_labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles[::-1], legend_labels[::-1], loc="upper left", frameon=True, markerscale=3)
        for h in legend.legend_handles:
            h.set_alpha(1.0)
        legend.get_frame().set_alpha(0.3)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"已保存 {OUTPUT_DIR / fname}")


if __name__ == "__main__":
    main()
