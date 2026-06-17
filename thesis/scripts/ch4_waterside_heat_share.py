"""Chapter 4 — 余热锅炉水侧吸热占比 (2-1HRSG1吸热占比.png, 2-1HRSG2吸热占比.png)"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1].parent / "Energy_Utilization_Analysis"
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
ST_POWER_COL = 37

HRSG_COLS = {
    1: {"run": 47, "hp_heat": 51, "ip_rh_heat": 52, "lp_heat": 53},
    2: {"run": 55, "hp_heat": 59, "ip_rh_heat": 60, "lp_heat": 61},
}


def is_running(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1"])


def build_hrsg_data(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    cols = HRSG_COLS[unit]
    data = pd.DataFrame({
        "st_power_mw": df.iloc[:, ST_POWER_COL] / 1e6,
        "hp_heat_mw": df.iloc[:, cols["hp_heat"]] / 1e6,
        "ip_rh_heat_mw": df.iloc[:, cols["ip_rh_heat"]] / 1e6,
        "lp_heat_mw": df.iloc[:, cols["lp_heat"]] / 1e6,
    })
    mask = (
        is_running(df.iloc[:, cols["run"]])
        & data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["st_power_mw"].between(0, 500)
        & (data["hp_heat_mw"] > 0)
        & (data["ip_rh_heat_mw"] > 0)
        & (data["lp_heat_mw"] > 0)
    )
    return data.loc[mask].sort_values("st_power_mw").reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for unit, fname in [(1, "2-1HRSG1吸热占比.png"), (2, "2-1HRSG2吸热占比.png")]:
        data = build_hrsg_data(df, unit)
        total = data["hp_heat_mw"] + data["ip_rh_heat_mw"] + data["lp_heat_mw"]
        hp_share = data["hp_heat_mw"] / total
        ip_share = hp_share + data["ip_rh_heat_mw"] / total
        lp_share = ip_share + data["lp_heat_mw"] / total

        fig, ax = plt.subplots(figsize=(5, 4.5))
        for y, label in [(hp_share, "高压主蒸汽吸热占比"), (ip_share, "中压+再热吸热占比"), (lp_share, "低压主蒸汽吸热占比")]:
            ax.scatter(data["st_power_mw"], y, s=POINT_SIZE, alpha=POINT_ALPHA, label=label, edgecolors="none")

        ax.set_title(f"HRSG{unit} 吸热占比")
        ax.set_xlabel("汽机出力 / MW")
        ax.set_ylabel("累计吸热占比")
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles[::-1], labels[::-1], loc="lower left", frameon=True, markerscale=3)
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
