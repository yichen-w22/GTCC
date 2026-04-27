import sys
import importlib.util
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GAS_MODULE_PATH = PROJECT_ROOT / "energy_analysis" / "working_fluid" / "gas.py"
GAS_SPEC = importlib.util.spec_from_file_location("working_fluid_gas", GAS_MODULE_PATH)
gas_module = importlib.util.module_from_spec(GAS_SPEC)
GAS_SPEC.loader.exec_module(gas_module)
calc_gas_density = gas_module.calc_gas_density


DATA_PATH = Path(
    r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\continuous_data_10min.csv"
)
OUTPUT_CSV_PATH = CURRENT_DIR / "gas_density_over_time.csv"
OUTPUT_PNG_PATH = CURRENT_DIR / "gas_density_over_time.png"
SAMPLE_INTERVAL_MINUTES = 10

FUEL_COMPONENT_COLUMNS = [
    "H2",
    "N2",
    "CO2",
    "CH4",
    "CO",
    "O2+Ar",
    "C2H6",
    "C3H8",
    "iC4H10",
    "nC4H10",
    "iC5H12",
    "nC5H12",
]


def build_time_axis(df):
    for column in df.columns:
        if any(token in str(column).lower() for token in ("time", "date", "timestamp", "日期", "时间", "时刻")):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().any():
                return parsed
    return pd.to_timedelta(df.index * SAMPLE_INTERVAL_MINUTES, unit="min")


def build_density_dataframe(df):
    time_axis = build_time_axis(df)
    records = []

    for idx, row in df.iterrows():
        composition = {column: row[column] for column in FUEL_COMPONENT_COLUMNS}
        records.append(
            {
                "sample_index": idx,
                "time": time_axis.iloc[idx] if hasattr(time_axis, "iloc") else time_axis[idx],
                "gas_name": "GT1 fuel",
                "temperature_K": row["燃料温度_1"],
                "pressure_Pa": row["燃料压力_1"],
                "density_kg_per_m3": calc_gas_density(
                    T=row["燃料温度_1"],
                    P=row["燃料压力_1"],
                    composition=composition,
                ),
            }
        )
        records.append(
            {
                "sample_index": idx,
                "time": time_axis.iloc[idx] if hasattr(time_axis, "iloc") else time_axis[idx],
                "gas_name": "GT2 fuel",
                "temperature_K": row["燃料温度_2"],
                "pressure_Pa": row["燃料压力_2"],
                "density_kg_per_m3": calc_gas_density(
                    T=row["燃料温度_2"],
                    P=row["燃料压力_2"],
                    composition=composition,
                ),
            }
        )

    return pd.DataFrame(records)


def save_plot(result_df):
    if plt is None:
        print("matplotlib is not installed; skipped plot generation")
        return

    plot_df = result_df.copy()
    if pd.api.types.is_timedelta64_dtype(plot_df["time"]):
        x = plot_df["time"].dt.total_seconds() / 3600.0
        x_label = "Time from start (hours)"
    else:
        x = plot_df["time"]
        x_label = "Time"

    fig, ax = plt.subplots(figsize=(12, 6))
    for gas_name in plot_df["gas_name"].unique():
        gas_slice = plot_df[plot_df["gas_name"] == gas_name]
        if pd.api.types.is_timedelta64_dtype(plot_df["time"]):
            x_values = gas_slice["time"].dt.total_seconds() / 3600.0
        else:
            x_values = gas_slice["time"]
        ax.plot(x_values, gas_slice["density_kg_per_m3"], label=gas_name, linewidth=1.4)

    ax.set_title("Fuel Gas Density vs Time")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density (kg/m^3)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"plot saved to: {OUTPUT_PNG_PATH}")


def print_summary(result_df):
    for gas_name in result_df["gas_name"].unique():
        gas_slice = result_df[result_df["gas_name"] == gas_name]
        print(
            f"{gas_name}: "
            f"min={gas_slice['density_kg_per_m3'].min():.6f} kg/m^3, "
            f"max={gas_slice['density_kg_per_m3'].max():.6f} kg/m^3, "
            f"mean={gas_slice['density_kg_per_m3'].mean():.6f} kg/m^3"
        )


def main():
    df = pd.read_csv(DATA_PATH)
    result_df = build_density_dataframe(df)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print_summary(result_df)
    print(f"data saved to: {OUTPUT_CSV_PATH}")
    save_plot(result_df)


if __name__ == "__main__":
    main()
