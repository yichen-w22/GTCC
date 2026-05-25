import sys
from pathlib import Path

import pandas as pd

# 这里是对气体组分组装的分析

PROJECT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_DIR / "Energy_Utilization_Analysis"
DATA_PATH = PACKAGE_ROOT / "data_precessing" / "continuous_data_10min.csv"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from energy_analysis import GasComposition, calc_fuel_lhv, calc_gas_density

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

MANUAL_COMPONENT_NAME_MAP = {
    "氢气": "H2",
    "氮气": "N2",
    "二氧化碳": "CO2",
    "甲烷": "CH4",
    "一氧化碳": "CO",
    "氧气": "O2+Ar",
    "氧气+氩气": "O2+Ar",
    "乙烷": "C2H6",
    "丙烷": "C3H8",
    "异丁烷": "iC4H10",
    "正丁烷": "nC4H10",
    "异戊烷": "iC5H12",
    "正戊烷": "nC5H12",
    "正己烷": None,
}

MANUAL_COMPOSITION_PERCENT = {
    "甲烷": 93.81,
    "乙烷": 2.46,
    "丙烷": 0.49,
    "异丁烷": 0.07,
    "正丁烷": 0.08,
    "异戊烷": 0.02,
    "正戊烷": 0.02,
    "正己烷": 0.01,
    "氮气": 1.21,
    "氧气": 0.02,
    "二氧化碳": 0.81,
}


def build_average_fuel_composition(df):
    average_components = df[FUEL_COMPONENT_COLUMNS].mean().to_dict()
    return GasComposition.from_dict(average_components).normalized()


def build_manual_fuel_composition(components, *, unit="percent", ignore_unsupported=True):
    composition_data = {}

    for name, value in components.items():
        species = MANUAL_COMPONENT_NAME_MAP.get(name, name)
        if species is None:
            if ignore_unsupported:
                continue
            raise ValueError(f"Unsupported component: {name}")

        composition_data[species] = composition_data.get(species, 0.0) + float(value)

    if unit == "percent":
        composition_data = {key: value / 100.0 for key, value in composition_data.items()}
    elif unit != "fraction":
        raise ValueError("unit must be 'percent' or 'fraction'")

    return GasComposition.from_dict(composition_data).normalized()


def print_composition(title, composition):
    print(title)
    for species, fraction in composition.as_dict().items():
        if fraction > 0:
            print(f"{species}: {fraction:.8f}")


def print_fuel_properties(title, composition, temperature, pressure):
    density = calc_gas_density(T=temperature, P=pressure, composition=composition)
    lhv_mass = calc_fuel_lhv(composition)
    lhv_volume = lhv_mass * density

    print(title)
    print(f"T = {temperature:.3f} K")
    print(f"P = {pressure:.3f} Pa")
    print(f"Density = {density:.6f} kg/m3")
    print(f"Lower heating value = {lhv_mass / 1e6:.6f} MJ/kg")
    print(f"Volumetric heating value = {lhv_volume / 1e6:.6f} MJ/m3")


df = pd.read_csv(DATA_PATH)

average_fuel_composition = build_average_fuel_composition(df)
manual_fuel_composition = build_manual_fuel_composition(MANUAL_COMPOSITION_PERCENT, unit="percent")

temperature = 293.15
pressure = 101325.0

print_composition("Average natural gas composition from CSV:", average_fuel_composition)
print()
print_fuel_properties("Average fuel properties from CSV:", average_fuel_composition, temperature, pressure)

print()
print_composition("Manual natural gas composition:", manual_fuel_composition)
print()
print_fuel_properties("Manual fuel properties:", manual_fuel_composition, temperature, pressure)
