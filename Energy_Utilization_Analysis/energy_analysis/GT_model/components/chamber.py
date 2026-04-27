import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass

from energy_analysis.ST_model.components.converter import EnergyConverter
from energy_analysis.working_fluid.gas import GasComposition, GasState, calc_fuel_lhv


@dataclass
class ChamberResult:
    state_3: GasState
    released_heat: float
    pressure_loss: float
    fuel_lhv: float


@dataclass
class Chamber(EnergyConverter):
    def state_3(
        self,
        total_pressure_recovery: float = 0.95,
        combustion_efficiency: float = 0.99,
        inlet_air: GasState | None = None,
        inlet_fuel: GasState | None = None,
        outlet_compositon: GasComposition | None = None,
    ) -> GasState:

        inlet_air = inlet_air
        inlet_fuel = inlet_fuel

        m_air = inlet_air.m_dot
        m_fuel = inlet_fuel.m_dot
        m_out = m_air + m_fuel
        fuel_lhv = calc_fuel_lhv(inlet_fuel.composition)
        released_heat = combustion_efficiency * fuel_lhv * m_fuel
        h_3 = (m_air * inlet_air.h + m_fuel * inlet_fuel.h + released_heat) / m_out
        p_3 = inlet_air.P * total_pressure_recovery

        state_3 = GasState.from_Ph(
            P=p_3,
            h=h_3,
            m_dot=m_out,
            composition=outlet_compositon.normalized(),
            name=f"{self.name}_state_3",
            ref=inlet_air.ref,
        )
        if self.outlets:
            self.outlets[0] = state_3
        else:
            self.add_outlet(state_3)
        return state_3

    def solve(
        self,
        inlet_air: GasState,
        inlet_fuel: GasState,
        outlet_compositon: GasComposition,
        total_pressure_recovery: float = 0.95,
        combustion_efficiency: float = 0.99,
    ) -> ChamberResult:
        state_3 = self.state_3(
            total_pressure_recovery=total_pressure_recovery,
            combustion_efficiency=combustion_efficiency,
            inlet_air=inlet_air,
            inlet_fuel=inlet_fuel,
            outlet_compositon=outlet_compositon,
        )
        fuel_lhv = calc_fuel_lhv(inlet_fuel.composition)
        released_heat = combustion_efficiency * fuel_lhv * inlet_fuel.m_dot
        return ChamberResult(
            state_3=state_3,
            released_heat=released_heat,
            pressure_loss=inlet_air.P - state_3.P,
            fuel_lhv=fuel_lhv,
        )
