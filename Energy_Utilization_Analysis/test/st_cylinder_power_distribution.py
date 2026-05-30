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

POINT_SIZE = 4
POINT_ALPHA = 0.30


def build_st_data(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "st_power_mw": df["ST.汽轮机功率"] / 1e6,
            "hp_power_mw": df["ST.高压缸出力"] / 1e6,
            "ip_power_mw": df["ST.中压缸出力"] / 1e6,
            "lp_power_mw": df["ST.低压缸出力（正算）"] / 1e6,
        }
    )

    mask = (
        data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["st_power_mw"].between(0, 300)
        & (data["hp_power_mw"] > 0)
        & (data["ip_power_mw"] > 0)
        & (data["lp_power_mw"] > 0)
    )

    data = data.loc[mask].copy()
    print(f"汽机: 绘图点数 {len(data)}")
    return data.sort_values("st_power_mw").reset_index(drop=True)


def draw_cumulative_scatter(ax, data: pd.DataFrame) -> None:
    hp_top = data["hp_power_mw"]
    ip_top = hp_top + data["ip_power_mw"]
    lp_top = ip_top + data["lp_power_mw"]

    layers = [
        (hp_top, "高压缸出力"),
        (ip_top, "中压缸出力"),
        (lp_top, "低压缸出力"),
    ]

    for y, label in layers:
        ax.scatter(
            data["st_power_mw"],
            y,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            label=label,
            edgecolors="none",
        )


def plot_st_power_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    data = build_st_data(df)
    draw_cumulative_scatter(ax, data)
    ax.set_xlabel("汽轮机总功率 / MW")
    ax.set_ylabel("累计出力 / MW")
    ax.set_ylim(0, 300)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles[::-1],
        labels[::-1],
        loc="upper left",
        frameon=True,
        markerscale=3,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    fig.tight_layout()
    plt.show()


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    plot_st_power_distribution(df)


if __name__ == "__main__":
    main()