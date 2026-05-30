# 这个是化简的物性计算函数
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional
import math

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.working_fluid.fluid import FlowState


R_UNIVERSAL = 8.314462618
T_REF = 298.15
P_REF = 101325.0


SPECIES = [
    "H2", "N2", "CO2", "CH4", "CO", "O2", "H2O",
    "C2H6", "C3H8", "iC4H10", "nC4H10", "iC5H12", "nC5H12",
]


MOLAR_MASS = {
    "H2": 2.016e-3,
    "N2": 28.0134e-3,
    "CO2": 44.0095e-3,
    "CH4": 16.043e-3,
    "CO": 28.0101e-3,
    "O2": 31.998e-3,
    "H2O": 18.0153e-3,
    "C2H6": 30.070e-3,
    "C3H8": 44.097e-3,
    "iC4H10": 58.123e-3,
    "nC4H10": 58.123e-3,
    "iC5H12": 72.150e-3,
    "nC5H12": 72.150e-3,
}


DEFAULT_DRY_AIR = {
    "O2": 0.21,
    "N2": 0.79,
}


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

    _molar_mass_cache: Optional[float] = field(default=None, init=False, repr=False)

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

        return GasComposition.from_dict({
            sp: x / total
            for sp, x in comp.items()
            if x > 0.0
        })

    def molar_mass(self) -> float:
        if self._molar_mass_cache is not None:
            return self._molar_mass_cache

        comp = self.as_dict()
        total = sum(comp.values())

        if total <= 0.0:
            raise ValueError("composition total must be positive")

        if abs(total - 1.0) > 1e-8:
            comp = {sp: x / total for sp, x in comp.items() if x > 0.0}

        self._molar_mass_cache = sum(
            x * MOLAR_MASS[sp]
            for sp, x in comp.items()
            if x > 0.0
        )

        return self._molar_mass_cache


@dataclass
class GasReferenceEnv:
    T0: float
    P0: float


def relative_humidity_to_mole_fraction(T: float, P: float, RH: float) -> float:
    """
    简化版本：
    为了减少物性库调用，默认忽略湿度影响。
    保留函数名和输入输出，保证外部代码不需要改。
    """
    return 0.0


def build_air_composition(T: float, P: float, RH: float) -> GasComposition:
    """
    简化空气组分：
    只保留 O2/N2，不考虑湿度、CO2、Ar。
    """
    return GasComposition.from_dict(DEFAULT_DRY_AIR).normalized()


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

    fuel = fuel_comp.as_dict()
    air = air_comp.as_dict()

    n_fuel = m_dot_fuel / fuel_comp.molar_mass()
    n_air = m_dot_air / air_comp.molar_mass()

    n_N2 = n_fuel * fuel.get("N2", 0.0) + n_air * air.get("N2", 0.0)

    n_CO2 = n_fuel * fuel.get("CO2", 0.0)
    n_H2O = n_fuel * fuel.get("H2O", 0.0)

    for sp, v in CARBON_ATOMS.items():
        n_CO2 += n_fuel * fuel.get(sp, 0.0) * v

    for sp, v in H2O_PRODUCTS.items():
        n_H2O += n_fuel * fuel.get(sp, 0.0) * v

    o2_in = n_fuel * fuel.get("O2", 0.0) + n_air * air.get("O2", 0.0)

    o2_need = 0.0
    for sp, v in O2_REQUIRED.items():
        o2_need += n_fuel * fuel.get(sp, 0.0) * v

    n_O2 = max(o2_in - o2_need, 0.0)

    return GasComposition.from_dict({
        "N2": n_N2,
        "O2": n_O2,
        "CO2": n_CO2,
        "H2O": n_H2O,
    }).normalized()


def _guess_gas_type(composition: GasComposition) -> str:
    comp = composition.normalized().as_dict()

    fuel_fraction = (
        comp.get("CH4", 0.0)
        + comp.get("C2H6", 0.0)
        + comp.get("C3H8", 0.0)
        + comp.get("iC4H10", 0.0)
        + comp.get("nC4H10", 0.0)
        + comp.get("iC5H12", 0.0)
        + comp.get("nC5H12", 0.0)
        + comp.get("H2", 0.0)
        + comp.get("CO", 0.0)
    )

    co2_h2o = comp.get("CO2", 0.0) + comp.get("H2O", 0.0)

    if fuel_fraction > 0.2:
        return "fuel"

    if co2_h2o > 0.03:
        return "flue_gas"

    return "air"


def _species_cp_mass(species: str, T: float) -> float:
    """
    返回单组分代表定压比热，单位 J/(kg·K)。
    这里是工程近似，不追求精细物性。
    """

    if species == "N2":
        if T < 500:
            return 1040.0
        elif T < 900:
            return 1100.0
        elif T < 1300:
            return 1180.0
        else:
            return 1240.0

    if species == "O2":
        if T < 500:
            return 920.0
        elif T < 900:
            return 980.0
        elif T < 1300:
            return 1050.0
        else:
            return 1120.0

    if species == "CO2":
        if T < 500:
            return 850.0
        elif T < 900:
            return 1050.0
        elif T < 1300:
            return 1200.0
        else:
            return 1300.0

    if species == "H2O":
        if T < 500:
            return 1860.0
        elif T < 900:
            return 2050.0
        elif T < 1300:
            return 2250.0
        else:
            return 2450.0

    if species == "CH4":
        if T < 500:
            return 2200.0
        elif T < 900:
            return 2600.0
        else:
            return 3100.0

    if species in ("C2H6", "C3H8", "iC4H10", "nC4H10", "iC5H12", "nC5H12"):
        if T < 500:
            return 1800.0
        elif T < 900:
            return 2400.0
        else:
            return 3000.0

    if species == "H2":
        return 14300.0

    if species == "CO":
        if T < 700:
            return 1040.0
        else:
            return 1150.0

    return 1100.0


def _mixture_cp_mass(T: float, composition: GasComposition) -> float:
    comp = composition.normalized().as_dict()

    mass_sum = 0.0
    cp_mass_sum = 0.0

    for sp, x in comp.items():
        if x <= 0.0:
            continue

        m_i = x * MOLAR_MASS[sp]
        mass_sum += m_i
        cp_mass_sum += m_i * _species_cp_mass(sp, T)

    return cp_mass_sum / mass_sum


def _representative_cp(T: float, composition: GasComposition) -> float:
    """
    先用组分加权 cp。
    这个函数仍是定比热近似，但 cp 会随当前温度区间变化。
    """
    return _mixture_cp_mass(T, composition)


def mixture_h_s_cp(T: float, P: float, composition):
    """
    保留原函数名和输出：
    return h, s, cp

    h: J/kg
    s: J/(kg·K)
    cp: J/(kg·K)

    简化假设：
    h = cp(T) * (T - T_REF)
    s = cp(T) * ln(T/T_REF) - R * ln(P/P_REF)
    """

    composition = composition.normalized()

    cp = _representative_cp(T, composition)
    R = R_UNIVERSAL / composition.molar_mass()

    h = cp * (T - T_REF)
    s = cp * math.log(T / T_REF) - R * math.log(P / P_REF)

    return h, s, cp


def calc_fuel_lhv(composition) -> float:
    comp = composition.normalized().as_dict()

    lhv_molar = 0.0

    for sp, lhv_mass in PURE_FUEL_LHV.items():
        lhv_molar += comp.get(sp, 0.0) * lhv_mass * MOLAR_MASS[sp]

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

        if abs(value_mid - target_value) <= tol * max(abs(target_value), 1.0):
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
            ref = GasReferenceEnv(T0=T_REF, P0=P_REF)
        else:
            ref = self.ref

        ref_state = self.from_TP(
            ref.T0,
            ref.P0,
            composition=self.composition,
            ref=ref
        )

        return (self.h - ref_state.h) - ref.T0 * (self.s - ref_state.s)
