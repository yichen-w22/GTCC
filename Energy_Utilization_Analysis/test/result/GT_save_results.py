import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.GT_model.GT_model import GTModel
from Energy_Utilization_Analysis.energy_analysis.working_fluid.streams import build_gases_from_row


DATA_PATH = PROJECT_ROOT / "data_precessing" / "continuous_data_10min.csv"
OUTPUT_PATH_GT1 = Path(__file__).resolve().parent / "gt1_results_wide.csv"
OUTPUT_PATH_GT2 = Path(__file__).resolve().parent / "gt2_results_wide.csv"

SAMPLE_SIZE = 10000
RANDOM_SEED = 42


def sample_dataframe(df, sample_size):
    if len(df) <= sample_size:
        sampled = df.copy()
    else:
        sampled = df.sample(n=sample_size, random_state=RANDOM_SEED).sort_index().copy()
    sampled["source_idx"] = sampled.index
    return sampled


def get_gt_states(df, idx, unit):
    gases = list(build_gases_from_row(df, idx).values())
    if unit == 1:
        return {
            "fuel": gases[0],
            "state_1": gases[2],
            "state_2": gases[4],
            "state_4": gases[6],
        }
    if unit == 2:
        return {
            "fuel": gases[1],
            "state_1": gases[3],
            "state_2": gases[5],
            "state_4": gases[7],
        }
    raise ValueError("unit must be 1 or 2")


def flatten_state(prefix, state):
    data = {
        f"{prefix}_name": state.name,
        f"{prefix}_T": state.T,
        f"{prefix}_P": state.P,
        f"{prefix}_m_dot": state.m_dot,
        f"{prefix}_h": state.h,
        f"{prefix}_s": state.s,
    }
    if hasattr(state, "cp"):
        data[f"{prefix}_cp"] = state.cp
    if hasattr(state, "R"):
        data[f"{prefix}_R"] = state.R
    return data


def flatten_result(source_idx, actual_power, result):
    row = {
        "source_idx": source_idx,
        "actual_power": actual_power,
        "net_power": result.net_power,
        "compressor_efficiency": result.compressor.efficiency,
        "compressor_delta_h": result.compressor.delta_h,
        "compressor_power": result.compressor.power,
        "chamber_released_heat": result.chamber.released_heat,
        "chamber_pressure_loss": result.chamber.pressure_loss,
        "chamber_fuel_lhv": result.chamber.fuel_lhv,
        "turbine_efficiency": result.turbine.efficiency,
        "turbine_delta_h": result.turbine.delta_h,
        "turbine_power": result.turbine.power,
    }

    row.update(flatten_state("state_1", result.state_1))
    row.update(flatten_state("state_2", result.state_2))
    row.update(flatten_state("fuel", result.fuel))
    row.update(flatten_state("state_3", result.state_3))
    row.update(flatten_state("state_4", result.state_4))
    row.update(flatten_state("state_2s", result.compressor.state_2s))
    row.update(flatten_state("state_4s", result.turbine.state_4s))

    if result.compressor.bleeding is not None:
        row.update(flatten_state("compressor_bleeding", result.compressor.bleeding))
    else:
        row.update(
            {
                "compressor_bleeding_name": None,
                "compressor_bleeding_T": None,
                "compressor_bleeding_P": None,
                "compressor_bleeding_m_dot": None,
                "compressor_bleeding_h": None,
                "compressor_bleeding_s": None,
                "compressor_bleeding_cp": None,
                "compressor_bleeding_R": None,
            }
        )

    return row


def collect_rows(df_raw, sampled_df, unit, gt_model):
    rows = []
    for _, sampled_row in sampled_df.iterrows():
        source_idx = int(sampled_row["source_idx"])
        actual_power = sampled_row[f"燃机出力_{unit}"]
        if pd.isna(actual_power):
            continue

        try:
            gt_states = get_gt_states(df_raw, source_idx, unit)
            result = gt_model.solve(
                state_1=gt_states["state_1"],
                state_2=gt_states["state_2"],
                fuel=gt_states["fuel"],
                state_4=gt_states["state_4"],
            )
        except Exception as exc:
            print(f"Skip GT{unit} idx={source_idx}, error={exc}")
            continue

        rows.append(flatten_result(source_idx, float(actual_power), result))

        if len(rows) % 20 == 0:
            print(f"Collected {len(rows)} valid GT{unit} results, current source_idx={source_idx}")

    return rows


def main():
    df_raw = pd.read_csv(DATA_PATH)
    sampled_df = sample_dataframe(df_raw, SAMPLE_SIZE)
    gt_model = GTModel()

    rows_gt1 = collect_rows(df_raw, sampled_df, 1, gt_model)
    rows_gt2 = collect_rows(df_raw, sampled_df, 2, gt_model)

    result_df_gt1 = pd.DataFrame(rows_gt1)
    result_df_gt2 = pd.DataFrame(rows_gt2)
    result_df_gt1.to_csv(OUTPUT_PATH_GT1, index=False, encoding="utf-8-sig")
    result_df_gt2.to_csv(OUTPUT_PATH_GT2, index=False, encoding="utf-8-sig")

    print(f"Raw data points = {len(df_raw)}")
    print(f"Sampled points = {len(sampled_df)}")
    print(f"Saved GT1 rows = {len(result_df_gt1)}")
    print(f"Saved GT2 rows = {len(result_df_gt2)}")
    print(f"GT1 output file = {OUTPUT_PATH_GT1}")
    print(f"GT2 output file = {OUTPUT_PATH_GT2}")


if __name__ == "__main__":
    main()
