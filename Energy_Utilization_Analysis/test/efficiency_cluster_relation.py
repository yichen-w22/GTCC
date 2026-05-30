from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"

POWER_LOWER_MW = 100.0
POWER_UPPER_MW = 285.0

PM_COLS = {
    1: {"run": 1, "power": 4, "comp_eta": 7, "turb_eta": 9},
    2: {"run": 19, "power": 22, "comp_eta": 25, "turb_eta": 27},
}

RAW_COLS = {
    1: {"compressor_inlet_t": 51},
    2: {"compressor_inlet_t": 0},
}


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1"])


def build_unit_data(pm: pd.DataFrame, raw: pd.DataFrame, unit: int) -> pd.DataFrame:
    pm_cols = PM_COLS[unit]
    raw_cols = RAW_COLS[unit]

    run_mask = is_true(pm.iloc[:, pm_cols["run"]])
    mask = (
        run_mask
        & (pm.iloc[:, pm_cols["comp_eta"]] > 0)
        & (pm.iloc[:, pm_cols["turb_eta"]] > 0)
        & (pm.iloc[:, pm_cols["power"]] / 1e6).between(POWER_LOWER_MW, POWER_UPPER_MW)
    )

    indexes = pm.loc[mask, "idx"].astype(int)
    return pd.DataFrame(
        {
            "idx": indexes.to_numpy(),
            "power_mw": pm.loc[mask].iloc[:, pm_cols["power"]].to_numpy(dtype=float) / 1e6,
            "comp_eta": pm.loc[mask].iloc[:, pm_cols["comp_eta"]].to_numpy(dtype=float),
            "turb_eta": pm.loc[mask].iloc[:, pm_cols["turb_eta"]].to_numpy(dtype=float),
            "compressor_inlet_t": raw.iloc[indexes.to_numpy(), raw_cols["compressor_inlet_t"]].to_numpy(dtype=float),
        }
    )


def plot_unit(unit: int, data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharex=True,
        sharey=True,
        gridspec_kw={"right": 0.88, "wspace": 0.16},
    )

    vmin = data["compressor_inlet_t"].quantile(0.01)
    vmax = data["compressor_inlet_t"].quantile(0.99)

    scatter0 = axes[0].scatter(
        data["power_mw"],
        data["comp_eta"],
        c=data["compressor_inlet_t"],
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=3,
        alpha=0.45,
    )
    axes[0].set_title(f"GT{unit} 压气机效率")
    axes[0].set_xlabel("燃机实际功率 / MW")
    axes[0].set_ylabel("压气机等熵效率")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        data["power_mw"],
        data["turb_eta"],
        c=data["compressor_inlet_t"],
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=3,
        alpha=0.45,
    )
    axes[1].set_title(f"GT{unit} 透平效率")
    axes[1].set_xlabel("燃机实际功率 / MW")
    axes[1].set_ylabel("透平等熵效率")
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_ylim(0.7, 1.0)

    cbar_ax = fig.add_axes([0.91, 0.16, 0.018, 0.70])
    cbar = fig.colorbar(scatter0, cax=cbar_ax)
    cbar.set_label("压气机进口温度 / K")


    fig.subplots_adjust(right=0.88, top=0.86, bottom=0.12, wspace=0.16)
    plt.show()


def main() -> None:
    pm = pd.read_csv(PLANT_METRICS_PATH)
    raw = pd.read_csv(RAW_DATA_PATH)

    for unit in (1, 2):
        data = build_unit_data(pm, raw, unit)
        print(
            f"GT{unit}: n={len(data)}, "
            f"T_in range={data['compressor_inlet_t'].min():.2f}-"
            f"{data['compressor_inlet_t'].max():.2f} K"
        )
        plot_unit(unit, data)


if __name__ == "__main__":
    main()
