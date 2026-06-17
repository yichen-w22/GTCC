import sys
from functools import cached_property
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass

from energy_analysis.ST_model.components.converter import EnergyConverter
from energy_analysis.working_fluid.gas import GasState


@dataclass
class Compressor(EnergyConverter):
    inlet_gas: GasState | None = None
    outlet_gas: GasState | None = None
    bleeding_mass_fraction: float = 0.0
    bleeding_pressure_fraction: float = 1.0
    bleeding_energy_fraction: float = 1.0

    @cached_property
    def state_2s(self) -> GasState:
        return self.inlet_gas.__class__.from_Ps(
            P=self.outlet_gas.P,
            s=self.inlet_gas.s,
            m_dot=self.outlet_gas.m_dot,
            composition=self.inlet_gas.composition,
            name=f"{self.name}_state_2s",
            ref=self.inlet_gas.ref,
        )

    @cached_property
    def isentropic_efficiency(self) -> float:
        return (self.state_2s.h - self.inlet_gas.h) / (self.outlet_gas.h - self.inlet_gas.h)

    @cached_property
    def exergy_efficiency(self) -> float:
        return (self.outlet_gas.exergy - self.inlet_gas.exergy) / (self.outlet_gas.h - self.inlet_gas.h)

    @cached_property
    def isentropic_loss(self) -> float | None:
        if self.outlet_gas.m_dot is None:
            return None
        return (self.outlet_gas.h - self.state_2s.h) * self.outlet_gas.m_dot

    @cached_property
    def bleeding(self) -> GasState:
        bleeding_mass = self.inlet_gas.m_dot * self.bleeding_mass_fraction
        bleeding_pressure = self.inlet_gas.P + (self.outlet_gas.P - self.inlet_gas.P) * self.bleeding_pressure_fraction
        bleeding_energy = self.inlet_gas.h + (self.outlet_gas.h - self.inlet_gas.h) * self.bleeding_energy_fraction
        return GasState.from_Ph(
            P=bleeding_pressure,
            h=bleeding_energy,
            m_dot=bleeding_mass,
            composition=self.inlet_gas.composition,
            name=f"{self.name}_bleeding",
            ref=self.inlet_gas.ref,
        )

    @cached_property
    def state_2(self) -> GasState:
        m_dot = self.inlet_gas.m_dot * (1.0 - self.bleeding_mass_fraction)
        return self.outlet_gas.__class__.from_TP(
            T=self.outlet_gas.T,
            P=self.outlet_gas.P,
            m_dot=m_dot,
            composition=self.outlet_gas.composition,
            name=f"{self.name}_state_2",
            ref=self.outlet_gas.ref,
        )

    @cached_property
    def power(self) -> float | None:
        if self.inlet_gas.m_dot is None or self.state_2.m_dot is None:
            return None
        return self.state_2.h * self.state_2.m_dot + self.bleeding.h * self.bleeding.m_dot - self.inlet_gas.h * self.inlet_gas.m_dot

    @cached_property
    def delta_h(self) -> float:
        return self.outlet_gas.h - self.inlet_gas.h
