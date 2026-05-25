from __future__ import annotations

import sys
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional

import CoolProp.CoolProp as CP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.working_fluid.fluid import FlowState

R_UNIVERSAL = 8.314462618
T_REF = 298.15
P_REF = 101325.0
P_SHOMATE_REF = 100000.0
MIN_GAS_PROPERTY_T = 200.0

# Shomate coefficients for ideal-gas water vapor, valid for the gas phase over
# the temperature range used by the GT/HRSG gas calculations.
H2O_SHOMATE_LOW = {
    "T_min": 298.0,
    "T_max": 1700.0,
    "A": 30.09200,
    "B": 6.832514,
    "C": 6.793435,
    "D": -2.534480,
    "E": 0.082139,
    "F": -250.8810,
    "G": 223.3967,
    "H": -241.8264,
}

H2O_SHOMATE_HIGH = {
    "T_min": 1700.0,
    "T_max": 6000.0,
    "A": 41.96426,
    "B": 8.622053,
    "C": -1.499780,
    "D": 0.098119,
    "E": -11.15764,
    "F": -272.1797,
    "G": 219.7809,
    "H": -241.8264,
}

SPECIES = [
    "H2", "N2", "CO2", "CH4", "CO", "O2", "H2O",
    "C2H6", "C3H8", "iC4H10", "nC4H10", "iC5H12", "nC5H12",
]

SPECIES_TO_COOLPROP = {
    "H2": "Hydrogen",
    "N2": "Nitrogen",
    "CO2": "CarbonDioxide",
    "CH4": "Methane",
    "CO": "CarbonMonoxide",
    "O2": "Oxygen",
    "H2O": "Water",
    "C2H6": "Ethane",
    "C3H8": "Propane",
    "iC4H10": "IsoButane",
    "nC4H10": "n-Butane",
    "iC5H12": "Isopentane",
    "nC5H12": "n-Pentane",
}

DEFAULT_DRY_AIR = {"O2": 0.2095, "N2": 0.7808, "CO2": 0.0004}

PURE_FUEL_LHV = {
    "H2": 119.96e6,
    "CH4": 50.01e6,
    "CO": 10.11e6,
    "C2H6": 47.50e6,
    "C3H8": 46.35e6,
    "iC4H10": 45.61e6,
    "nC4H10": 45.75e6,
    "iC5H12": 45.24e6,
    "nC5H12": 45.36e6,
}

CARBON_ATOMS = {
    "CH4": 1.0,
    "CO": 1.0,
    "C2H6": 2.0,
    "C3H8": 3.0,
    "iC4H10": 4.0,
    "nC4H10": 4.0,
    "iC5H12": 5.0,
    "nC5H12": 5.0,
}

H2O_PRODUCTS = {
    "H2": 1.0,
    "CH4": 2.0,
    "C2H6": 3.0,
    "C3H8": 4.0,
    "iC4H10": 5.0,
    "nC4H10": 5.0,
    "iC5H12": 6.0,
    "nC5H12": 6.0,
}

O2_REQUIRED = {
    "H2": 0.5,
    "CO": 0.5,
    "CH4": 2.0,
    "C2H6": 3.5,
    "C3H8": 5.0,
    "iC4H10": 6.5,
    "nC4H10": 6.5,
    "iC5H12": 8.0,
    "nC5H12": 8.0,
}


@dataclass
class GasComposition:
    H2: float = 0.0
    N2: float = 0.0
    CO2: float = 0.0
    CH4: float = 0.0
    CO: float = 0.0
    O2: float = 0.0
    H2O: float = 0.0
    C2H6: float = 0.0
    C3H8: float = 0.0
    iC4H10: float = 0.0
    nC4H10: float = 0.0
    iC5H12: float = 0.0
    nC5H12: float = 0.0

    @classmethod
    def from_dict(cls, composition: Dict[str, float]) -> "GasComposition":
        data = {sp: float(composition.get(sp, 0.0)) for sp in SPECIES}
        data["O2"] += float(composition.get("O2+Ar", 0.0))
        data["O2"] += float(composition.get("O2_Ar", 0.0))
        return cls(**data)

    def as_dict(self) -> Dict[str, float]:
        return {sp: getattr(self, sp) for sp in SPECIES}

    def normalized(self) -> "GasComposition":
        comp = self.as_dict()
        total = sum(comp.values())
        if total <= 0.0:
            raise ValueError("composition total must be positive")
        return GasComposition.from_dict({sp: x / total for sp, x in comp.items()})

    def molar_mass(self) -> float:
        comp = self.normalized().as_dict()
        return sum(x * CP.PropsSI("M", SPECIES_TO_COOLPROP[sp]) for sp, x in comp.items())


@dataclass
class GasReferenceEnv:
    T0: float
    P0: float


def relative_humidity_to_mole_fraction(T: float, P: float, RH: float) -> float:
    RH = RH / 100.0
    p_sat = CP.PropsSI("P", "T", T, "Q", 0, "Water")
    return RH * p_sat / P


def build_air_composition(T: float, P: float, RH: float) -> GasComposition:
    x_h2o = relative_humidity_to_mole_fraction(T, P, RH)
    comp = {sp: x * (1.0 - x_h2o) for sp, x in DEFAULT_DRY_AIR.items()}
    comp["H2O"] = x_h2o
    return GasComposition.from_dict(comp).normalized()


def build_flue_gas_composition(
    fuel_composition,
    air_composition,
    m_dot_fuel: float,
    m_dot_air: float,
) -> GasComposition:
    m_dot_fuel = max(float(m_dot_fuel), 0.0)
    m_dot_air = max(float(m_dot_air), 0.0)

    if m_dot_fuel == 0.0 and m_dot_air == 0.0:
        return air_composition.normalized()

    fuel_comp = fuel_composition.normalized()
    air_comp = air_composition.normalized()

    n_fuel = m_dot_fuel / fuel_comp.molar_mass()
    n_air = m_dot_air / air_comp.molar_mass()

    fuel = fuel_comp.as_dict()
    air = air_comp.as_dict()
    flue = {sp: 0.0 for sp in SPECIES}

    for sp in ("N2", "O2", "CO2", "H2O"):
        flue[sp] = n_fuel * fuel[sp] + n_air * air[sp]

    flue["CO2"] += n_fuel * sum(fuel[sp] * v for sp, v in CARBON_ATOMS.items())
    flue["H2O"] += n_fuel * sum(fuel[sp] * v for sp, v in H2O_PRODUCTS.items())

    o2_in = n_fuel * fuel["O2"] + n_air * air["O2"]
    o2_need = n_fuel * sum(fuel[sp] * v for sp, v in O2_REQUIRED.items())
    flue["O2"] = max(o2_in - o2_need, 0.0)

    if sum(flue.values()) <= 0.0:
        return air_composition.normalized()

    return GasComposition.from_dict(flue).normalized()


def _pure_gas_molar_h_s_cp(species: str, T: float, P: float):
    if species == "H2O":
        return _water_vapor_molar_h_s_cp(T, P)

    fluid = SPECIES_TO_COOLPROP[species]
    return (
        CP.PropsSI("HMOLAR", "T", T, "P", P, fluid),
        CP.PropsSI("SMOLAR", "T", T, "P", P, fluid),
        CP.PropsSI("CPMOLAR", "T", T, "P", P, fluid),
    )


def _water_vapor_molar_h_s_cp(T: float, P: float):
    coeff = H2O_SHOMATE_LOW if T < H2O_SHOMATE_HIGH["T_min"] else H2O_SHOMATE_HIGH
    return _shomate_molar_h_s_cp(T, P, coeff)


def _shomate_molar_h_s_cp(T: float, P: float, coeff: dict):
    t = T / 1000.0

    A = coeff["A"]
    B = coeff["B"]
    C = coeff["C"]
    D = coeff["D"]
    E = coeff["E"]
    F = coeff["F"]
    G = coeff["G"]
    H = coeff["H"]

    cp = A + B * t + C * t**2 + D * t**3 + E / t**2
    h = A * t + B * t**2 / 2.0 + C * t**3 / 3.0 + D * t**4 / 4.0 - E / t + F - H
    s = A * math.log(t) + B * t + C * t**2 / 2.0 + D * t**3 / 3.0 - E / (2.0 * t**2) + G
    s -= R_UNIVERSAL * math.log(P / P_SHOMATE_REF)

    return h * 1000.0, s, cp


def mixture_h_s_cp(T: float, P: float, composition):
    composition = composition.normalized()
    hmolar = smolar = cpmolar = 0.0
    hmolar_ref = smolar_ref = 0.0

    for species, x in composition.as_dict().items():
        if x <= 0.0:
            continue
        p_i = x * P
        p_i_ref = x * P_REF
        h_i, s_i, cp_i = _pure_gas_molar_h_s_cp(species, T, p_i)
        h_i_ref, s_i_ref, _ = _pure_gas_molar_h_s_cp(species, T_REF, p_i_ref)

        hmolar += x * h_i
        smolar += x * s_i
        cpmolar += x * cp_i
        hmolar_ref += x * h_i_ref
        smolar_ref += x * s_i_ref

    m_mix = composition.molar_mass()
    return (
        (hmolar - hmolar_ref) / m_mix,
        (smolar - smolar_ref) / m_mix,
        cpmolar / m_mix,
    )


def calc_fuel_lhv(composition) -> float:
    comp = composition.normalized().as_dict()

    lhv_molar = sum(
        comp[sp] * lhv_mass * CP.PropsSI("M", SPECIES_TO_COOLPROP[sp])
        for sp, lhv_mass in PURE_FUEL_LHV.items()
    )
    return lhv_molar / composition.molar_mass()

def calc_gas_density(T: float, P: float, composition) -> float:
    composition = composition.normalized()
    return P * composition.molar_mass() / (R_UNIVERSAL * T)

def solve_temperature_from_property(
    P: float,
    target_value: float,
    composition,
    property_name: str,
    T_low: float = 250.0,
    T_high: float = 2500.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:

    composition = composition.normalized()
    T_low = max(T_low, MIN_GAS_PROPERTY_T)
    prop_index = 0 if property_name == "h" else 1

    def calc_property(T: float) -> float:
        return mixture_h_s_cp(T, P, composition)[prop_index]

    value_low = calc_property(T_low)
    value_high = calc_property(T_high)

    while target_value < value_low and T_low > MIN_GAS_PROPERTY_T:
        T_low = max(MIN_GAS_PROPERTY_T, T_low - 20.0)
        value_low = calc_property(T_low)

    while target_value > value_high:
        T_high += 100.0
        value_high = calc_property(T_high)

    if not value_low <= target_value <= value_high:
        raise ValueError("target property is out of search range")

    for _ in range(max_iter):
        T_mid = 0.5 * (T_low + T_high)
        value_mid = calc_property(T_mid)

        if abs(value_mid - target_value) <= tol * target_value:
            return T_mid
        if value_mid < target_value:
            T_low = T_mid
        else:
            T_high = T_mid

    return 0.5 * (T_low + T_high)


def create_gas_reference_env(T0=298.15, P0=101325.0) -> GasReferenceEnv:
    return GasReferenceEnv(T0=T0, P0=P0)


@dataclass
class GasState(FlowState):
    composition: GasComposition = field(default_factory=GasComposition)
    cp: Optional[float] = None
    R: Optional[float] = None

    @classmethod
    def from_TP(cls, T, P, m_dot=None, composition=None, name="", ref=None):
        composition = composition.normalized()
        h, s, cp = mixture_h_s_cp(T, P, composition)
        return cls(
            name=name,
            T=T,
            P=P,
            m_dot=m_dot,
            h=h,
            s=s,
            ref=ref,
            composition=composition,
            cp=cp,
            R=R_UNIVERSAL / composition.molar_mass(),
        )

    @classmethod
    def from_Ph(cls, P, h, m_dot=None, composition=None, name="", ref=None):
        composition = composition.normalized()
        T = solve_temperature_from_property(P, h, composition, "h")
        return cls.from_TP(T, P, m_dot=m_dot, composition=composition, name=name, ref=ref)

    @classmethod
    def from_Ps(cls, P, s, m_dot=None, composition=None, name="", ref=None):
        composition = composition.normalized()
        T = solve_temperature_from_property(P, s, composition, "s")
        return cls.from_TP(T, P, m_dot=m_dot, composition=composition, name=name, ref=ref)

    @cached_property
    def exergy(self):
        if self.ref is None:
            ref_env = GasReferenceEnv(T0=T_REF, P0=P_REF)
        else:
            ref_env = self.ref

        ref = self.from_TP(ref_env.T0, ref_env.P0, composition=self.composition, ref=ref_env)
        ref_h = ref.h
        ref_s = ref.s
        return (self.h - ref_h) - ref_env.T0 * (self.s - ref_s)
