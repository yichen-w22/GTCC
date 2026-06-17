"""Chapter 4 — 汽轮机出力计算结果 (3汽机出力计算.png)"""

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

POWER_OFFSET_MW = 0.99 * 0.98


def print_error_summary(actual_mw: pd.Series, calculated_mw: pd.Series) -> None:
    residual_mw = calculated_mw - actual_mw
    relative_error = residual_mw / actual_mw
    print(f"样本数：{len(actual_mw)}")
    print(f"平均残差：{residual_mw.mean():.4f} MW")
    print(f"平均绝对误差：{residual_mw.abs().mean():.4f} MW")
    print(f"均方根误差：{((residual_mw**2).mean()) ** 0.5:.4f} MW")
    print(f"平均百分比偏差：{relative_error.mean() * 100:.4f}%")
    print(f"平均绝对百分比误差：{relative_error.abs().mean() * 100:.4f}%")
    print(f"相对均方根误差：{((relative_error**2).mean()) ** 0.5 * 100:.4f}%")
    print(f"百分比误差中位数：{relative_error.median() * 100:.4f}%")


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    upper = 150 * 1e6
    mask = (df["ST.汽轮机计算功率"] > 0) & (df["ST.汽轮机计算功率"] < upper)
    actual_mw = df["ST.汽轮机功率"][mask] / 1e6
    calculated_mw = df["ST.汽轮机计算功率"][mask] / 1e6 * POWER_OFFSET_MW

    print_error_summary(actual_mw, calculated_mw)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, upper / 1e6], [0, upper / 1e6], "r--", linewidth=1.2, alpha=0.7)
    ax.plot(actual_mw, calculated_mw, ".", markersize=2, alpha=0.9)
    ax.set_xlabel("汽轮机实际功率 / MW")
    ax.set_ylabel("汽轮机计算功率 / MW")
    ax.set_xlim(40, 150)
    ax.set_ylim(40, 150)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "3汽机出力计算.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"已保存 {OUTPUT_DIR / '3汽机出力计算.png'}")


if __name__ == "__main__":
    main()
