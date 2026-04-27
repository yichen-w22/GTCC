import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass

from energy_analysis.ST_model.components.converter import EnergyConverter
from energy_analysis.working_fluid.gas import GasState


@dataclass
class TurbineResult:
    state_4: GasState
    state_4s: GasState
    efficiency: float
    delta_h: float
    power: float | None


@dataclass
class Turbine(EnergyConverter):
    def isentropic_efficiency(self, state_3: GasState, state_4: GasState) -> tuple[float, GasState]:
        state_4s = state_3.__class__.from_Ps(
            P=state_4.P,
            s=state_3.s,
            m_dot=state_4.m_dot,
            composition=state_3.composition,
            name=f"{self.name}_state_4s",
            ref=state_3.ref,
        )
        efficiency = (state_3.h - state_4.h) / (state_3.h - state_4s.h)
        return efficiency, state_4s

    def work(self, state_3: GasState, state_4: GasState, bleeding: GasState | None = None) -> float | None:
        if state_3.m_dot is None:
            return None
        return state_3.m_dot * state_3.h + bleeding.m_dot * bleeding.h - state_4.m_dot * state_4.h
    
    def solve(
        self,
        state_3: GasState,
        state_4: GasState,
        bleeding: GasState | None = None,
    ) -> TurbineResult:
        efficiency, state_4s = self.isentropic_efficiency(state_3, state_4)
        power = self.work(state_3, state_4, bleeding=bleeding)
        return TurbineResult(
            state_4=state_4,
            state_4s=state_4s,
            efficiency=efficiency,
            delta_h=state_3.h - state_4.h,
            power=power
        )
