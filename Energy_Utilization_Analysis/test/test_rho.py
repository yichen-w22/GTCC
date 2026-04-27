import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Energy_Utilization_Analysis.energy_analysis.working_fluid.streams import build_fuel_composition_from_row
from energy_analysis.working_fluid.gas import GasComposition
from energy_analysis.working_fluid.gas import R_UNIVERSAL


DATA_PATH = Path(
    r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\continuous_data_10min.csv"
)

T_STD = 273.15
P_STD = 101325.0


def calc_standard_density(composition, T_std=T_STD, P_std=P_STD):
    if isinstance(composition, dict):
        composition = GasComposition.from_dict(composition)
    molar_mass = composition.molar_mass()
    return P_std * molar_mass / (R_UNIVERSAL * T_std)


def main():
    df = pd.read_csv(DATA_PATH)
    rho_values = []

    for idx in range(len(df)):
        composition = build_fuel_composition_from_row(df, idx)
        rho_std = calc_standard_density(composition)
        rho_values.append(rho_std)

    rho_avg = sum(rho_values) / len(rho_values)

    print(f"sample_count = {len(rho_values)}")
    print(f"T_std = {T_STD} K")
    print(f"P_std = {P_STD} Pa")
    print(f"average_standard_density = {rho_avg:.6f} kg/m^3")


if __name__ == "__main__":
    main()
