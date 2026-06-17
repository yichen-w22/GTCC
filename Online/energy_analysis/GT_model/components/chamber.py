import sys
from functools import cached_property
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass

from energy_analysis.ST_model.components.converter import EnergyConverter
from energy_analysis.working_fluid.gas import GasComposition, GasState, calc_fuel_lhv


@dataclass
class Chamber(EnergyConverter):
    inlet_air: GasState | None = None
    inlet_fuel: GasState | None = None
    outlet_composition: GasComposition | None = None
    total_pressure_recovery: float = 0.95
    combustion_efficiency: float = 0.99
    bleeding: GasState | None = None

    @cached_property
    def fuel_lhv(self) -> float:
        return calc_fuel_lhv(self.inlet_fuel.composition)

    @cached_property
    def released_heat(self) -> float:
        return self.combustion_efficiency * self.fuel_lhv * self.inlet_fuel.m_dot

    @cached_property
    def state_3(self) -> GasState: # 燃烧室出口状态
        m_air = self.inlet_air.m_dot
        m_fuel = self.inlet_fuel.m_dot
        m_out = m_air + m_fuel
        h_3 = (m_air * self.inlet_air.h + m_fuel * self.inlet_fuel.h + self.released_heat) / m_out
        p_3 = self.inlet_air.P * self.total_pressure_recovery

        state_3 = GasState.from_Ph(
            P=p_3,
            h=h_3,
            m_dot=m_out,
            composition=self.outlet_composition.normalized(),
            name=f"{self.name}_state_3",
            ref=self.inlet_air.ref,
        )
        return state_3

    @cached_property
    def state_3_c(self) -> GasState: # 等效冷却后的燃烧室出口/透平入口状态
        m_air = self.inlet_air.m_dot
        m_fuel = self.inlet_fuel.m_dot
        m_bleeding = self.bleeding.m_dot
        m_out_c = m_air + m_fuel + m_bleeding
        h_3_c = (m_air * self.inlet_air.h + m_fuel * self.inlet_fuel.h + self.released_heat + m_bleeding * self.bleeding.h) / m_out_c
        p_3_c = self.inlet_air.P * self.total_pressure_recovery

        state_3_c = GasState.from_Ph(
            P=p_3_c,
            h=h_3_c,
            m_dot=m_out_c,
            composition=self.outlet_composition.normalized(),
            name=f"{self.name}_state_3_c",
            ref=self.inlet_air.ref,
        )
        return state_3_c

    @cached_property
    def pressure_loss(self) -> float:
        return self.inlet_air.P - self.state_3.P
