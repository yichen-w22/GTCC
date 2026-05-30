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
POINT_ALPHA = 0.35

COLS = {
    1: {
        "run": 47,
        "flue_flow": 2,
        "heat_efficiency": 48,
        "effectiveness": 49,
        "exergy_efficiency": 50,
    },
    2: {
        "run": 55,
        "flue_flow": 20,
        "heat_efficiency": 56,
        "effectiveness": 57,
        "exergy_efficiency": 58,
    },
}

METRICS = [
    ("effectiveness", "换热器有效性"),
    ("heat_efficiency", "换热效率"),
    ("exergy_efficiency", "㶲效率"),
]


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1"])


def build_unit_data(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    cols = COLS[unit]
    data = pd.DataFrame(
        {
            "flue_flow": df.iloc[:, cols["flue_flow"]],
            "heat_efficiency": df.iloc[:, cols["heat_efficiency"]],
            "effectiveness": df.iloc[:, cols["effectiveness"]],
            "exergy_efficiency": df.iloc[:, cols["exergy_efficiency"]],
        }
    )

    mask = (
        is_true(df.iloc[:, cols["run"]])
        & data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["flue_flow"].between(100, 1000)
        & data["heat_efficiency"].between(0.5, 1.05)
        & data["effectiveness"].between(0.5, 1.05)
        & data["exergy_efficiency"].between(0.5, 1.05)
    )
    data = data.loc[mask].copy()
    print(f"HRSG{unit}: 绘图点数 {len(data)}")

    return data.sort_values("flue_flow").reset_index(drop=True)


def draw_metric(ax, data: pd.DataFrame, metric: str, color: str, label: str) -> None:
    ax.scatter(
        data["flue_flow"],
        data[metric],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        color=color,
        label=label,
        edgecolors="none",
    )


def plot_hrsg_performance(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 7.2),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.16, "hspace": 0.28},
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for unit in (1, 2):
        data = build_unit_data(df, unit)
        color = colors[unit - 1]
        for ax, (metric, title) in zip(axes[unit - 1], METRICS):
            draw_metric(ax, data, metric, color, f"HRSG{unit}")
            ax.set_title(title)
            ax.set_xlim(300, 650)
            ax.set_ylim(0.7, 1.05)
            ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("烟气流量 / (kg/s)")

    axes[0, 0].set_ylabel("HRSG1\n性能指标")
    axes[1, 0].set_ylabel("HRSG2\n性能指标")

    fig.suptitle("余热锅炉性能随烟气流量的变化")
    fig.tight_layout()
    plt.show()


def main() -> None:
    df = pd.read_csv(PLANT_METRICS_PATH)
    plot_hrsg_performance(df)


if __name__ == "__main__":
    main()
