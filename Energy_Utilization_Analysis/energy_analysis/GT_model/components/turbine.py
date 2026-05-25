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
class Turbine(EnergyConverter):
    state_3_c: GasState | None = None
    state_4: GasState | None = None

    @cached_property
    def state_4s(self) -> GasState:
        return self.state_3_c.__class__.from_Ps(
            P=self.state_4.P,
            s=self.state_3_c.s,
            m_dot=self.state_4.m_dot,
            composition=self.state_3_c.composition,
            name=f"{self.name}_state_4s",
            ref=self.state_3_c.ref,
        )

    @cached_property
    def isentropic_efficiency(self) -> float:
        return (self.state_3_c.h - self.state_4.h) / (self.state_3_c.h - self.state_4s.h)

    @cached_property
    def efficiency(self) -> float:
        return self.isentropic_efficiency

    @cached_property
    def exergy_efficiency(self) -> float:
        return (self.state_3_c.exergy - self.state_4.exergy) / (self.state_3_c.exergy - self.state_4s.exergy)

    @cached_property
    def isentropic_loss(self) -> float | None:
        if self.state_4.m_dot is None:
            return None
        return (self.state_4.h - self.state_4s.h) * self.state_4.m_dot

    @cached_property
    def power(self) -> float | None:
        if self.state_3_c.m_dot is None:
            return None
        return self.state_3_c.m_dot * self.state_3_c.h - self.state_4.m_dot * self.state_4.h
