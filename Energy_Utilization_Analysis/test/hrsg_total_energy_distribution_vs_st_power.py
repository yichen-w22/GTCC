from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
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
CP_FLUE_GAS = 1120.0

ST_POWER_COL = 37

RAW_COLS = {
    "ambient_2": 10,
    "inlet_temp_2": 47,
    "outlet_temp_2": 50,
    "ambient_1": 61,
    "inlet_temp_1": 98,
    "outlet_temp_1": 101,
    "flue_flow_2": 132,
    "flue_flow_1": 133,
}


def sensible_heat_power(m_dot, temperature, reference_temperature):
    return m_dot * CP_FLUE_GAS * (temperature - reference_temperature)


def build_distribution_data(raw: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    selected_raw = raw.iloc[metrics["idx"].astype(int)].reset_index(drop=True)
    ambient_temperature = selected_raw.iloc[
        :, [RAW_COLS["ambient_1"], RAW_COLS["ambient_2"]]
    ].mean(axis=1)

    inlet_energy = (
        sensible_heat_power(
            selected_raw.iloc[:, RAW_COLS["flue_flow_1"]],
            selected_raw.iloc[:, RAW_COLS["inlet_temp_1"]],
            ambient_temperature,
        )
        + sensible_heat_power(
            selected_raw.iloc[:, RAW_COLS["flue_flow_2"]],
            selected_raw.iloc[:, RAW_COLS["inlet_temp_2"]],
            ambient_temperature,
        )
    )
    stack_loss = (
        sensible_heat_power(
            selected_raw.iloc[:, RAW_COLS["flue_flow_1"]],
            selected_raw.iloc[:, RAW_COLS["outlet_temp_1"]],
            ambient_temperature,
        )
        + sensible_heat_power(
            selected_raw.iloc[:, RAW_COLS["flue_flow_2"]],
            selected_raw.iloc[:, RAW_COLS["outlet_temp_2"]],
            ambient_temperature,
        )
    )
    generation = metrics.iloc[:, ST_POWER_COL]
    condenser = inlet_energy - stack_loss - generation

    data = pd.DataFrame(
        {
            "st_power_mw": generation / 1e6,
            "stack_loss_mw": stack_loss / 1e6,
            "generation_mw": generation / 1e6,
            "condenser_mw": condenser / 1e6,
        }
    )

    mask = (
        data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["st_power_mw"].between(40, 300)
        & (inlet_energy > 50e6)
        & (data["stack_loss_mw"] > 0)
        & (data["generation_mw"] > 0)
        & (data["condenser_mw"] > 0)
    )

    data = data.loc[mask].sort_values("st_power_mw").reset_index(drop=True)
    print(f"绘图点数 {len(data)}")
    return data


def draw_cumulative_scatter(ax, data: pd.DataFrame) -> None:
    stack_loss_top = data["stack_loss_mw"]
    generation_top = stack_loss_top + data["generation_mw"]
    condenser_top = generation_top + data["condenser_mw"]

    layers = [
        (stack_loss_top, "余热锅炉出口尾气损失"),
        (generation_top, "汽机发电"),
        (condenser_top, "凝汽器放热 + 热网供热"),
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


def plot_hrsg_total_energy_distribution(df_raw: pd.DataFrame, df_metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    data = build_distribution_data(df_raw, df_metrics)
    draw_cumulative_scatter(ax, data)

    ax.set_xlabel("汽机出力 / MW")
    ax.set_ylabel("累计能量 / MW")
    ax.set_xlim(40, 300)
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
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_metrics = pd.read_csv(PLANT_METRICS_PATH)
    plot_hrsg_total_energy_distribution(df_raw, df_metrics)


if __name__ == "__main__":
    main()
