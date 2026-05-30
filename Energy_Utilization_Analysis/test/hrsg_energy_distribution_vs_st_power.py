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

ST_POWER_COL = 37

HRSG_COLS = {
    1: {
        "run": 47,
        "hp_heat": 51,
        "ip_rh_heat": 52,
        "lp_heat": 53,
    },
    2: {
        "run": 55,
        "hp_heat": 59,
        "ip_rh_heat": 60,
        "lp_heat": 61,
    },
}


def is_running(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1"])


def build_hrsg_data(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    cols = HRSG_COLS[unit]
    data = pd.DataFrame(
        {
            "st_power_mw": df.iloc[:, ST_POWER_COL] / 1e6,
            "hp_heat_mw": df.iloc[:, cols["hp_heat"]] / 1e6,
            "ip_rh_heat_mw": df.iloc[:, cols["ip_rh_heat"]] / 1e6,
            "lp_heat_mw": df.iloc[:, cols["lp_heat"]] / 1e6,
        }
    )

    mask = (
        is_running(df.iloc[:, cols["run"]])
        & data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["st_power_mw"].between(0, 500)
        & (data["hp_heat_mw"] > 0)
        & (data["ip_rh_heat_mw"] > 0)
        & (data["lp_heat_mw"] > 0)
    )

    data = data.loc[mask].copy()
    print(f"HRSG{unit}: 绘图点数 {len(data)}")
    return data.sort_values("st_power_mw").reset_index(drop=True)


def draw_cumulative_scatter(ax, data: pd.DataFrame) -> None:
    hp_top = data["hp_heat_mw"]
    ip_rh_top = hp_top + data["ip_rh_heat_mw"]
    lp_top = ip_rh_top + data["lp_heat_mw"]

    layers = [
        (hp_top, "高压主蒸汽吸热"),
        (ip_rh_top, "中压+再热吸热"),
        (lp_top, "低压主蒸汽吸热"),
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


def plot_hrsg_energy_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, unit in zip(axes, (1, 2)):
        data = build_hrsg_data(df, unit)
        draw_cumulative_scatter(ax, data)
        ax.set_title(f"HRSG{unit} 能量分配")
        ax.set_xlabel("汽机出力 / MW")
        ax.set_ylim(0, 400)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("累计吸热功率 / MW")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = axes[0].legend(
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
    plot_hrsg_energy_distribution(df)


if __name__ == "__main__":
    main()
