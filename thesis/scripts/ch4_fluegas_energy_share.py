"""Chapter 4 — 余热锅炉烟气侧能量占比 (1-1烟气能量占比.png)"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1].parent / "Energy_Utilization_Analysis"
RAW_DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "chapter4"

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

POINT_SIZE = 4
POINT_ALPHA = 0.30
CP_FLUE_GAS = 1120.0
ST_POWER_COL = 37

RAW_COLS = {
    "ambient_2": 10, "inlet_temp_2": 47, "outlet_temp_2": 50,
    "ambient_1": 61, "inlet_temp_1": 98, "outlet_temp_1": 101,
    "flue_flow_2": 132, "flue_flow_1": 133,
}


def sensible_heat_power(m_dot, temperature, ref_temperature):
    return m_dot * CP_FLUE_GAS * (temperature - ref_temperature)


def build_distribution_data(raw: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    sel = raw.iloc[metrics["idx"].astype(int)].reset_index(drop=True)
    amb = sel.iloc[:, [RAW_COLS["ambient_1"], RAW_COLS["ambient_2"]]].mean(axis=1)

    inlet_energy = (
        sensible_heat_power(sel.iloc[:, RAW_COLS["flue_flow_1"]], sel.iloc[:, RAW_COLS["inlet_temp_1"]], amb)
        + sensible_heat_power(sel.iloc[:, RAW_COLS["flue_flow_2"]], sel.iloc[:, RAW_COLS["inlet_temp_2"]], amb)
    )
    stack_loss = (
        sensible_heat_power(sel.iloc[:, RAW_COLS["flue_flow_1"]], sel.iloc[:, RAW_COLS["outlet_temp_1"]], amb)
        + sensible_heat_power(sel.iloc[:, RAW_COLS["flue_flow_2"]], sel.iloc[:, RAW_COLS["outlet_temp_2"]], amb)
    )
    generation = metrics.iloc[:, ST_POWER_COL]
    condenser = inlet_energy - stack_loss - generation

    data = pd.DataFrame({
        "st_power_mw": generation / 1e6,
        "stack_loss_mw": stack_loss / 1e6,
        "generation_mw": generation / 1e6,
        "condenser_mw": condenser / 1e6,
    })
    mask = (
        data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["st_power_mw"].between(40, 300)
        & (inlet_energy > 50e6)
        & (data["stack_loss_mw"] > 0)
        & (data["generation_mw"] > 0)
        & (data["condenser_mw"] > 0)
    )
    return data.loc[mask].sort_values("st_power_mw").reset_index(drop=True)


def main() -> None:
    raw = pd.read_csv(RAW_DATA_PATH)
    metrics = pd.read_csv(PLANT_METRICS_PATH)
    data = build_distribution_data(raw, metrics)

    total = data["stack_loss_mw"] + data["generation_mw"] + data["condenser_mw"]
    gen_share = data["generation_mw"] / total
    stack_share = gen_share + data["stack_loss_mw"] / total
    cond_share = stack_share + data["condenser_mw"] / total

    fig, ax = plt.subplots(figsize=(6, 4.5))
    layers = [
        (gen_share, "汽机发电占比"),
        (stack_share, "汽机发电 + 余热锅炉出口尾气损失占比"),
        (cond_share, "烟气入口总能量占比"),
    ]
    for y, label in layers:
        ax.scatter(data["st_power_mw"], y, s=POINT_SIZE, alpha=POINT_ALPHA, label=label, edgecolors="none")

    ax.set_xlabel("汽机出力 / MW")
    ax.set_ylabel("累计能量占比")
    ax.set_xlim(40, 300)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], loc="upper left", frameon=True, markerscale=3)
    for h in legend.legend_handles:
        h.set_alpha(1.0)
    legend.get_frame().set_alpha(0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "1-1烟气能量占比.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"已保存 {OUTPUT_DIR / '1-1烟气能量占比.png'}")


if __name__ == "__main__":
    main()
