"""Chapter 4 — 汽轮机效率随出力变化 (5高压缸效率.png, 5中压缸效率.png, 5低压缸效率.png)"""

from pathlib import Path

import matplotlib.pyplot as plt
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


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)

    cylinders = [
        ("ST.高压缸出力", "ST.高压缸等熵效率", "高压缸等熵效率", "5高压缸效率.png"),
        ("ST.中压缸出力", "ST.中压缸等熵效率", "中压缸等熵效率", "5中压缸效率.png"),
        ("ST.低压缸出力（正算）", "ST.低压缸等熵效率", "低压缸等熵效率", "5低压缸效率.png"),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for power_col, eff_col, ylabel, fname in cylinders:
        mask = df[power_col] > 0
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(df["ST.汽轮机功率"][mask] / 1e6, df[eff_col][mask], ".", markersize=2, alpha=0.5)
        ax.set_xlabel("汽轮机功率 / MW")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"已保存 {OUTPUT_DIR / fname}")


if __name__ == "__main__":
    main()
