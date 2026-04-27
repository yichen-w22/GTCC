from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import CoolProp.CoolProp as CP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.working_fluid.fluid import FlowState

R_UNIVERSAL = 8.314462618

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

    return GasComposition.from_dict(flue).normalized()


def _pure_gas_molar_h_s_cp(species: str, T: float, P: float):
    fluid = SPECIES_TO_COOLPROP[species]
    return (
        CP.PropsSI("HMOLAR", "T", T, "P", P, fluid),
        CP.PropsSI("SMOLAR", "T", T, "P", P, fluid),
        CP.PropsSI("CPMOLAR", "T", T, "P", P, fluid),
    )


def mixture_h_s_cp(T: float, P: float, composition):
    composition = composition.normalized()
    hmolar = smolar = cpmolar = 0.0

    for species, x in composition.as_dict().items():
        if x <= 0.0:
            continue
        p_i = x * P
        h_i, s_i, cp_i = _pure_gas_molar_h_s_cp(species, T, p_i)

        hmolar += x * h_i
        smolar += x * s_i
        cpmolar += x * cp_i

    m_mix = composition.molar_mass()
    return hmolar / m_mix, smolar / m_mix, cpmolar / m_mix


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
    prop_index = 0 if property_name == "h" else 1

    def calc_property(T: float) -> float:
        return mixture_h_s_cp(T, P, composition)[prop_index]

    value_low = calc_property(T_low)
    value_high = calc_property(T_high)

    while target_value < value_low and T_low > 200.0:
        T_low = max(200.0, T_low - 20.0)
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

    @property
    def exergy(self):
        ref = self.from_TP(self.ref.T0, self.ref.P0, composition=self.composition, ref=self.ref)
        ref_h = ref.h
        ref_s = ref.s
        return (self.h - ref_h) - self.ref.T0 * (self.s - ref_s)
