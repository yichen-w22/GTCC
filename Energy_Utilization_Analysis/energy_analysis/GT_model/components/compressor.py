import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass

from energy_analysis.ST_model.components.converter import EnergyConverter
from energy_analysis.working_fluid.gas import GasState


@dataclass
class CompressorResult:
    state_2: GasState
    state_2s: GasState
    efficiency: float
    delta_h: float
    power: float | None
    bleeding: GasState | None


@dataclass
class Compressor(EnergyConverter):
    def isentropic_efficiency(self, state_1: GasState, state_2: GasState) -> tuple[float, GasState]:
        state_2s = state_1.__class__.from_Ps(
            P=state_2.P,
            s=state_1.s,
            m_dot=state_2.m_dot,
            composition=state_1.composition,
            name=f"{self.name}_state_2s",
            ref=state_1.ref,
        )
        efficiency = (state_2s.h - state_1.h) / (state_2.h - state_1.h)
        return efficiency, state_2s

    def work(
        self,
        state_1: GasState,
        state_2: GasState,
        bleeding: GasState | None = None,
    ) -> float | None:
        if state_1.m_dot is None or state_2.m_dot is None:
            return None

        power = state_2.h * state_2.m_dot + bleeding.h * bleeding.m_dot - state_1.h * state_1.m_dot
        return power

    def bleeding(
        self,
        bleeding_mass_fraction: float,
        bleeding_pressure_fraction: float,
        bleeding_energy_fraction: float,
        inlet_gas: GasState,
        outlet_gas: GasState,
    ) -> GasState | None:

        bleeding_mass = inlet_gas.m_dot * bleeding_mass_fraction
        bleeding_pressure = inlet_gas.P + (outlet_gas.P - inlet_gas.P) * bleeding_pressure_fraction
        bleeding_energy = inlet_gas.h + (outlet_gas.h - inlet_gas.h) * bleeding_energy_fraction
        return GasState.from_Ph(
            P=bleeding_pressure,
            h=bleeding_energy,
            m_dot=bleeding_mass,
            composition=inlet_gas.composition,
            name=f"{self.name}_bleeding",
            ref=inlet_gas.ref,
        )

    def state_2(self, bleeding_mass_fraction: float, inlet_gas: GasState, outlet_gas: GasState) -> GasState:
        m_dot = inlet_gas.m_dot * (1.0 - bleeding_mass_fraction)
        return outlet_gas.__class__.from_TP(
            T=outlet_gas.T,
            P=outlet_gas.P,
            m_dot=m_dot,
            composition=outlet_gas.composition,
            name=f"{self.name}_state_2",
            ref=outlet_gas.ref,
        )

    def solve(
        self,
        inlet_gas: GasState,
        outlet_gas: GasState,
        bleeding_mass_fraction: float = 0.0,
        bleeding_pressure_fraction: float = 1.0,
        bleeding_energy_fraction: float = 1.0,
    ) -> CompressorResult:
        efficiency, state_2s = self.isentropic_efficiency(inlet_gas, outlet_gas)
        bleeding_gas = self.bleeding(
            bleeding_mass_fraction=bleeding_mass_fraction,
            bleeding_pressure_fraction=bleeding_pressure_fraction,
            bleeding_energy_fraction=bleeding_energy_fraction,
            inlet_gas=inlet_gas,
            outlet_gas=outlet_gas,
        )
        state_2 = self.state_2(bleeding_mass_fraction=bleeding_mass_fraction, inlet_gas=inlet_gas, outlet_gas=outlet_gas)
        power = self.work(state_1=inlet_gas, state_2=state_2, bleeding=bleeding_gas)
        delta_h = outlet_gas.h - inlet_gas.h
        return CompressorResult(
            state_2=state_2,
            state_2s=state_2s,
            efficiency=efficiency,
            delta_h=delta_h,
            power=power,
            bleeding=bleeding_gas,
        )
