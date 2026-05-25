import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from energy_analysis.ST_model.plant import build_plant
import numpy as np

df = pd.read_csv(r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\continuous_data_10min.csv")

mass_balance_ = []
for idx in range (1, 8000, 500):
    plant = build_plant(df, idx)
    mass_balance_.append(plant["hp_turbine"].mass_balance())

mass_balance = np.mean(mass_balance_)
print(mass_balance)