"""Chapter 3 — 燃气轮机出力计算值与实测值对比 (1燃机1出力.png, 1燃机2出力.png)"""

from pathlib import Path

import matplotlib.pyplot as plt
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

UPPER_MW = 300
LOWER_MW = 0


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

    configs = [
        ("GT1.燃机实际功", "GT1.燃机计算功", temp1, "GT1 实际功率 / MW", "GT1 计算功率 / MW", "1燃机1出力.png"),
        ("GT2.燃机实际功", "GT2.燃机计算功", temp2, "GT2 实际功率 / MW", "GT2 计算功率 / MW", "1燃机2出力.png"),
    ]

    mask1 = (df["GT1.燃机计算功"] > 0) & (df["GT1.燃机计算功"] < UPPER_MW * 1e6)
    mask2 = (df["GT2.燃机计算功"] > 0) & (df["GT2.燃机计算功"] < UPPER_MW * 1e6)
    masks = [mask1, mask2]

    vmin = min(temp1[mask1].quantile(0.01), temp2[mask2].quantile(0.01))
    vmax = max(temp1[mask1].quantile(0.99), temp2[mask2].quantile(0.99))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, (xcol, ycol, temp, xlab, ylab, fname) in enumerate(configs):
        m = masks[i]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sc = ax.scatter(
            df[xcol][m] / 1e6,
            df[ycol][m] / 1e6,
            c=temp[m],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=4,
            alpha=0.45,
            edgecolors="none",
        )
        ax.plot([LOWER_MW, UPPER_MW], [LOWER_MW, UPPER_MW], "r--", linewidth=1.2)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
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


if __name__ == "__main__":
    main()
