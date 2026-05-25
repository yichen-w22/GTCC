import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.GT_model.GT_model import GTModel
from energy_analysis.working_fluid.gas import (
    GasComposition,
    GasState,
    build_air_composition,
    build_flue_gas_composition,
    create_gas_reference_env,
)


DATA_PATH = PROJECT_ROOT / "data_precessing" / "continuous_data_10min.csv"

SENSITIVITY_SAMPLE_SIZE = 6
SENSITIVITY_RANDOM_SEED = 20260507
CORR_COFF_MIN = 0.90
CORR_COFF_MAX = 1.5
CORR_COFF_POINTS = 41
SENSITIVITY_OUTPUT_DIR = PROJECT_ROOT / "test" / "result" / "gt_air_flow_sensitivity"


def unit_columns(unit: int) -> dict[str, str]:
    return {
        "air_flow": f"燃机进口空气流量_{unit}",
        "fuel_flow": f"燃料质量流量_{unit}",
        "actual_power": f"燃机出力_{unit}",
        "fuel_T": f"燃料温度_{unit}",
        "fuel_P": f"燃料压力_{unit}",
        "ambient_T": f"环境温度_{unit}",
        "ambient_P": f"压气机入口压力_{unit}",
        "ambient_RH": f"大气相对湿度_{unit}",
        "compressor_out_T": f"压气机出口温度_{unit}",
        "compressor_out_P": f"压气机出口压力_{unit}",
        "flue_T": f"进口烟温_{unit}",
        "flue_P": f"进口烟压_{unit}",
    }


def fuel_composition_from_row(row: pd.Series) -> GasComposition:
    return GasComposition.from_dict(
        {
            "H2": row["H2"],
            "N2": row["N2"],
            "CO2": row["CO2"],
            "CH4": row["CH4"],
            "CO": row["CO"],
            "O2+Ar": row["O2+Ar"],
            "C2H6": row["C2H6"],
            "C3H8": row["C3H8"],
            "iC4H10": row["iC4H10"],
            "nC4H10": row["nC4H10"],
            "iC5H12": row["iC5H12"],
            "nC5H12": row["nC5H12"],
        }
    )


def reference_env_from_row(row: pd.Series):
    return create_gas_reference_env(
        T0=(row["环境温度_1"] + row["环境温度_2"]) / 2,
        P0=(row["压气机入口压力_1"] + row["压气机入口压力_2"]) / 2,
    )


def build_gt_result(row: pd.Series, unit: int, air_flow: float) -> GTModel:
    cols = unit_columns(unit)
    fuel_flow = row[cols["fuel_flow"]]

    gas_ref = reference_env_from_row(row)
    air_composition = build_air_composition(
        T=row[cols["ambient_T"]],
        P=row[cols["ambient_P"]],
        RH=row[cols["ambient_RH"]],
    )
    fuel_composition = fuel_composition_from_row(row)
    flue_gas_composition = build_flue_gas_composition(
        fuel_composition=fuel_composition,
        air_composition=air_composition,
        m_dot_fuel=fuel_flow,
        m_dot_air=air_flow,
    )

    fuel = GasState.from_TP(
        T=row[cols["fuel_T"]],
        P=row[cols["fuel_P"]],
        m_dot=fuel_flow,
        composition=fuel_composition,
        name=f"{unit}号炉燃料",
        ref=gas_ref,
    )
    state_1 = GasState.from_TP(
        T=row[cols["ambient_T"]],
        P=row[cols["ambient_P"]],
        m_dot=air_flow,
        composition=air_composition,
        name=f"{unit}号燃机入口空气",
        ref=gas_ref,
    )
    state_2 = GasState.from_TP(
        T=row[cols["compressor_out_T"]],
        P=row[cols["compressor_out_P"]],
        m_dot=air_flow,
        composition=air_composition,
        name=f"{unit}号燃机压气机出口",
        ref=gas_ref,
    )
    state_4 = GasState.from_TP(
        T=row[cols["flue_T"]],
        P=row[cols["flue_P"]],
        m_dot=fuel_flow + air_flow,
        composition=flue_gas_composition,
        name=f"{unit}号余热锅炉入口烟气",
        ref=gas_ref,
    )

    return GTModel(state_1=state_1, compressor_outlet=state_2, fuel=fuel, state_4=state_4)


def valid_sensitivity_indices(df: pd.DataFrame, unit: int) -> np.ndarray:
    cols = unit_columns(unit)
    required_columns = [
        cols["air_flow"],
        cols["fuel_flow"],
        cols["actual_power"],
        cols["fuel_T"],
        cols["fuel_P"],
        cols["ambient_T"],
        cols["ambient_P"],
        cols["ambient_RH"],
        cols["compressor_out_T"],
        cols["compressor_out_P"],
        cols["flue_T"],
        cols["flue_P"],
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
    valid = df[required_columns].notna().all(axis=1)
    valid &= df[cols["air_flow"]] > 0
    valid &= df[cols["fuel_flow"]] > 0
    valid &= df[cols["actual_power"]] != 0
    return df.index[valid].to_numpy()


def scan_air_flow_sensitivity(
    df: pd.DataFrame,
    unit: int,
    sample_indices: np.ndarray,
    corr_coffs: np.ndarray,
) -> pd.DataFrame:
    records = []
    cols = unit_columns(unit)

    for sample_no, idx in enumerate(sample_indices, start=1):
        row = df.loc[int(idx)].copy()
        original_air_flow = row[cols["air_flow"]]
        actual_power = row[cols["actual_power"]]

        for corr_coff in corr_coffs:
            try:
                corrected_air_flow = original_air_flow * corr_coff
                result = build_gt_result(row, unit, corrected_air_flow)
                calculated_power = result.net_power
                power_residual = calculated_power - actual_power
                records.append(
                    {
                        "unit": unit,
                        "sample_no": sample_no,
                        "idx": int(idx),
                        "corr_coff": corr_coff,
                        "air_flow": corrected_air_flow,
                        "original_air_flow": original_air_flow,
                        "actual_power": actual_power,
                        "calculated_power": calculated_power,
                        "power_residual": power_residual,
                        "power_residual_ratio": power_residual / actual_power,
                        "turbine_efficiency": result.turbine.efficiency,
                    }
                )
            except Exception as exc:
                print(f"GT{unit} idx={idx} corr_coff={corr_coff:.4f} sensitivity failed: {exc}")

    return pd.DataFrame(records)


def plot_air_flow_sensitivity(scan_df: pd.DataFrame, unit: int, output_dir: Path) -> None:
    if scan_df.empty:
        print(f"GT{unit} sensitivity result is empty, skip plotting.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for idx, group in scan_df.groupby("idx", sort=False):
        label = f"idx={idx}"
        axes[0].plot(group["corr_coff"], group["power_residual"] / 1e6, marker="o", linewidth=1.4, label=label)
        axes[1].plot(group["corr_coff"], group["turbine_efficiency"], marker="o", linewidth=1.4, label=label)

    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("power residual / MW")
    axes[0].set_title(f"GT{unit}: power residual vs air-flow correction coefficient")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("air-flow correction coefficient")
    axes[1].set_ylabel("turbine efficiency")
    axes[1].set_title(f"GT{unit}: turbine efficiency vs air-flow correction coefficient")
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    output_path = output_dir / f"gt{unit}_air_flow_sensitivity_curves.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"GT{unit} sensitivity plot saved: {output_path}")


def run_air_flow_sensitivity(df: pd.DataFrame, unit: int) -> None:
    SENSITIVITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    valid_indices = valid_sensitivity_indices(df, unit)
    if len(valid_indices) == 0:
        print(f"GT{unit} has no valid sensitivity samples.")
        return

    rng = np.random.default_rng(SENSITIVITY_RANDOM_SEED + unit)
    sample_size = min(SENSITIVITY_SAMPLE_SIZE, len(valid_indices))
    sample_indices = np.sort(rng.choice(valid_indices, size=sample_size, replace=False))
    corr_coffs = np.linspace(CORR_COFF_MIN, CORR_COFF_MAX, CORR_COFF_POINTS)

    print(f"GT{unit} random sensitivity sample indices: {sample_indices.tolist()}")
    scan_df = scan_air_flow_sensitivity(df, unit, sample_indices, corr_coffs)

    output_csv = SENSITIVITY_OUTPUT_DIR / f"gt{unit}_air_flow_sensitivity.csv"
    scan_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"GT{unit} sensitivity data saved: {output_csv}")
    plot_air_flow_sensitivity(scan_df, unit, SENSITIVITY_OUTPUT_DIR)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    for unit in (1, 2):
        run_air_flow_sensitivity(df, unit)

