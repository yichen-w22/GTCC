import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.GT_model.GT_model import GTModel
from Energy_Utilization_Analysis.energy_analysis.working_fluid.streams import build_gases_from_row


DATA_PATH = PROJECT_ROOT / "data_precessing" / "continuous_data_10min.csv"
ROW_RANGE = range(1000, 1400, 20)


def get_gt_states(df: pd.DataFrame, idx: int, unit: int) -> dict:
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


def check_gt_result(idx: int, unit: int, result, actual_power: float | None = None) -> list[str]:
    issues: list[str] = []

    if result.state_2.P <= result.state_1.P:
        issues.append("compressor outlet pressure does not increase")
    if result.state_2.T <= result.state_1.T:
        issues.append("compressor outlet temperature does not increase")
    if result.state_3.P >= result.state_2.P:
        issues.append("combustor pressure loss is not observed")
    if result.state_3.h <= result.state_2.h:
        issues.append("combustor enthalpy does not increase")
    if result.state_4.P >= result.state_3.P:
        issues.append("turbine outlet pressure does not decrease")
    if result.state_4.h >= result.state_3.h:
        issues.append("turbine enthalpy does not decrease")

    if result.compressor.efficiency <= 0.0 or result.compressor.efficiency > 1.05:
        issues.append(f"compressor efficiency out of range: {result.compressor.efficiency:.4f}")
    if result.turbine.efficiency <= 0.0 or result.turbine.efficiency > 1.05:
        issues.append(f"turbine efficiency out of range: {result.turbine.efficiency:.4f}")

    if result.compressor.power is None or result.compressor.power <= 0.0:
        issues.append("compressor power is not positive")
    if result.turbine.power is None or result.turbine.power <= 0.0:
        issues.append("turbine power is not positive")
    if result.net_power is None:
        issues.append("net power is None")

    if actual_power is not None and result.net_power is not None:
        relative_error = (result.net_power - actual_power) / actual_power if actual_power != 0 else float("nan")
        if pd.notna(relative_error) and abs(relative_error) > 0.30:
            issues.append(f"net power mismatch > 30%: relative_error={relative_error:.3f}")

    if issues:
        issues.insert(0, f"GT{unit} idx={idx}")
    return issues


def run_gt_diagnosis(df: pd.DataFrame, row_range) -> tuple[list[dict], list[str], list[str]]:
    gt_model = GTModel()
    ok_rows: list[dict] = []
    failed_rows: list[str] = []
    issue_rows: list[str] = []

    for idx in row_range:
        for unit in (1, 2):
            actual_power_column = f"燃机出力_{unit}"
            actual_power = df.iloc[idx][actual_power_column] if actual_power_column in df.columns else None

            try:
                gt_states = get_gt_states(df, idx, unit)
                result = gt_model.solve(
                    state_1=gt_states["state_1"],
                    state_2=gt_states["state_2"],
                    fuel=gt_states["fuel"],
                    state_4=gt_states["state_4"],
                )
            except Exception as exc:
                failed_rows.append(f"GT{unit} idx={idx} failed: {type(exc).__name__}: {exc}")
                continue

            issues = check_gt_result(idx, unit, result, actual_power=actual_power)
            if issues:
                issue_rows.append(" | ".join(issues))

            ok_rows.append(
                {
                    "idx": idx,
                    "unit": unit,
                    "actual_power": actual_power,
                    "net_power": result.net_power,
                    "compressor_efficiency": result.compressor.efficiency,
                    "turbine_efficiency": result.turbine.efficiency,
                    "state_1_T": result.state_1.T,
                    "state_2_T": result.state_2.T,
                    "state_3_T": result.state_3.T,
                    "state_4_T": result.state_4.T,
                }
            )

    return ok_rows, failed_rows, issue_rows


def print_summary(ok_rows: list[dict], failed_rows: list[str], issue_rows: list[str]) -> None:
    result_df = pd.DataFrame(ok_rows)

    print("GTModel diagnosis summary")
    print("-" * 60)
    print(f"success_cases = {len(ok_rows)}")
    print(f"failed_cases = {len(failed_rows)}")
    print(f"issue_cases = {len(issue_rows)}")

    if not result_df.empty:
        print("-" * 60)
        print("numeric summary")
        print(
            result_df[
                ["net_power", "compressor_efficiency", "turbine_efficiency", "state_1_T", "state_2_T", "state_3_T", "state_4_T"]
            ]
            .describe()
            .round(4)
        )

    if failed_rows:
        print("-" * 60)
        print("failed cases")
        for message in failed_rows[:10]:
            print(message)

    if issue_rows:
        print("-" * 60)
        print("issue cases")
        for message in issue_rows[:15]:
            print(message)
    else:
        print("-" * 60)
        print("no obvious GTModel issues found in sampled rows")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    ok_rows, failed_rows, issue_rows = run_gt_diagnosis(df, ROW_RANGE)
    print_summary(ok_rows, failed_rows, issue_rows)


if __name__ == "__main__":
    main()
