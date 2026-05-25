import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.ST_model.ST_model import STModel


DATA_PATH = PROJECT_ROOT / "data_precessing" / "continuous_data_10min.csv"
ROW_RANGE = range(1000, 1400, 20)


df = pd.read_csv(DATA_PATH)
st_model = STModel()

idx = 1000
result = st_model.solve(idx=idx, data_path=DATA_PATH)

lp_result = result.component_results["lp_turbine"]

for key, value in vars(lp_result).items():
    print(f"{key}: {value}")


