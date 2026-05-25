import sys
from pathlib import Path

# 这里是对汽机出力的分析
PROJECT_ROOT = Path(__file__).resolve().parent / "Energy_Utilization_Analysis"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass
from typing import Any

import pandas as pd

from energy_analysis import build_plant_from_streams, build_gases_from_row, build_streams_from_row

df = pd.read_csv(r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\continuous_data_10min.csv")

idx = 5000
streams = build_streams_from_row(df, idx)
gases = build_gases_from_row(df, idx)

plant = build_plant_from_streams(streams, gases)

st_power_real = df["汽机出力"].loc[idx]
print(st_power_real / 1e6)
hp_power = plant["hp_turbine"].power_output
ip_power = plant["ip_turbine"].power_output
lp_power = plant["lp_turbine"].power_output
lp_eff = plant["ip_turbine"].isentropic_efficiency
st_power_cal = hp_power + ip_power + lp_power
print(st_power_cal / 1e6)
print((st_power_real - st_power_cal) / 1e6)
print(lp_eff)