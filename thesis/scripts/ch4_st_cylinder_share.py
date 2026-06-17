"""Chapter 4 — 汽轮机各缸出力占比 (4-1各缸出力占比.png)"""

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


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    data = pd.DataFrame({
        "st_power_mw": df["ST.汽轮机功率"] / 1e6,
        "hp_power_mw": df["ST.高压缸出力"] / 1e6,
        "ip_power_mw": df["ST.中压缸出力"] / 1e6,
        "lp_power_mw": df["ST.低压缸出力（正算）"] / 1e6,
    })
    mask = (
        data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["st_power_mw"].between(0, 300)
        & (data["hp_power_mw"] > 0)
        & (data["ip_power_mw"] > 0)
        & (data["lp_power_mw"] > 0)
    )
    data = data.loc[mask].sort_values("st_power_mw").reset_index(drop=True)

    total = data["hp_power_mw"] + data["ip_power_mw"] + data["lp_power_mw"]
    hp_share = data["hp_power_mw"] / total
    ip_share = hp_share + data["ip_power_mw"] / total
    lp_share = ip_share + data["lp_power_mw"] / total

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for y, label in [(hp_share, "高压缸出力占比"), (ip_share, "中压缸出力占比"), (lp_share, "低压缸出力占比")]:
        ax.scatter(data["st_power_mw"], y, s=POINT_SIZE, alpha=POINT_ALPHA, label=label, edgecolors="none")

    ax.set_xlabel("汽轮机总功率 / MW")
    ax.set_ylabel("累计出力占比")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], loc="upper left", frameon=True, markerscale=3)
    for h in legend.legend_handles:
        h.set_alpha(1.0)
    legend.get_frame().set_alpha(0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "4-1各缸出力占比.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"已保存 {OUTPUT_DIR / '4-1各缸出力占比.png'}")


if __name__ == "__main__":
    main()
