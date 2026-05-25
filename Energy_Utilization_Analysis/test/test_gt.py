import sys
import time
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.GT_model.GT_model import build_gt


DATA_PATH = PROJECT_ROOT / "data_precessing" / "continuous_data_10min.csv"
OUTPUT_PATH = CURRENT_DIR / "gt_test_results.csv"
ROW_RANGE = range(1, 8000, 50)
UNIT = 1

# def collect_results(df, row_range, unit: int):
#     rows = []
#     for idx in row_range:
#         result = solve_gt_model(df, idx, unit)
#         actual_power = df.iloc[idx][f"燃机出力_{unit}"]
#         net_power_mw = result.net_power / 1e6
#         actual_power_mw = actual_power / 1e6
#         power_residual_mw = net_power_mw - actual_power_mw

#         rows.append(
#                 {
#                     "idx": idx,
#                     "net_power_MW": net_power_mw,
#                     "actual_power_MW": actual_power_mw,
#                     "power_residual_MW": power_residual_mw,
#                     "relative_error": power_residual_mw / actual_power_mw if actual_power_mw else None,
#                     "turbine_efficiency": result.turbine.efficiency,
#                     "compressor_efficiency": result.compressor.efficiency,
#                     "state_1_T_C": result.state_1.T - 273.15,
#                     "state_2_T_C": result.state_2.T - 273.15,
#                     "state_3_T_C": result.state_3.T - 273.15,
#                     "state_4_T_C": result.state_4.T - 273.15,
#                     "state_1_m_dot": result.state_1.m_dot,
#                     "fuel_m_dot": result.fuel.m_dot,
#                     "state_4_m_dot": result.state_4.m_dot,
#                     "delta_h_4_1_MJ_per_kg": (result.state_4.h - result.state_1.h) / 1e6,
#                     "error": None,
#                 }
#             )
#     return pd.DataFrame(rows)


# def main():
#     start = time.perf_counter()
#     df = pd.read_csv(DATA_PATH)
#     result_df = collect_results(df, ROW_RANGE, UNIT)
#     # result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
#     print(f"elapsed: {time.perf_counter() - start:.3f} s")

# main()

# df = pd.read_csv(OUTPUT_PATH)

# # plt.scatter(df["actual_power_MW"], df["state_3_T_C"])
# # plt.show()

# plt.scatter(df["actual_power_MW"], df["state_4T_C"])

# plt.show()

# df = pd.read_csv(DATA_PATH)
# idx = 2000
# unit = 1

# result = solve_gt_model(df, idx, unit)
# print(result.state_4.P)

df = pd.read_csv(DATA_PATH)
idx = 6000
unit = 2
result = build_gt(df, idx, unit)

print(result.turbine.efficiency)
