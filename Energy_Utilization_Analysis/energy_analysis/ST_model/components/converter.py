from dataclasses import dataclass, field
from functools import cached_property
from typing import List

from energy_analysis.working_fluid.fluid import FlowState


@dataclass
class EnergyConverterResult:
    inlets: List[FlowState] = field(default_factory=list)
    outlets: List[FlowState] = field(default_factory=list)

    @cached_property
    def inlet(self):
        return self.inlets[0] if self.inlets else None

    @cached_property
    def outlet(self):
        return self.outlets[0] if self.outlets else None


@dataclass
class EnergyConverter:
    name: str
    inlets: List[FlowState] = field(default_factory=list)
    outlets: List[FlowState] = field(default_factory=list)

    def add_inlet(self, stream: FlowState):
        self.inlets.append(stream)

    def add_outlet(self, stream: FlowState):
        self.outlets.append(stream)

    @staticmethod
    def _sum_defined(values):
        if any(value is None for value in values):
            return None
        return sum(values)

    @cached_property
    def mass_in(self):
        return self._sum_defined([s.m_dot for s in self.inlets])

    @cached_property
    def mass_out(self):
        return self._sum_defined([s.m_dot for s in self.outlets])

    @cached_property
    def energy_in(self):
        return self._sum_defined([s.energy_flow for s in self.inlets])

    @cached_property
    def energy_out(self):
        return self._sum_defined([s.energy_flow for s in self.outlets])

    @cached_property
    def exergy_in(self):
        return self._sum_defined([s.exergy_flow for s in self.inlets])

    @cached_property
    def exergy_out(self):
        return self._sum_defined([s.exergy_flow for s in self.outlets])

    @cached_property
    def mass_balance(self):
        mass_in = self.mass_in
        mass_out = self.mass_out
        if mass_in is None or mass_out is None:
            return None
        return mass_in - mass_out

    @cached_property
    def energy_balance(self):
        energy_in = self.energy_in
        energy_out = self.energy_out
        if energy_in is None or energy_out is None:
            return None
        return energy_in - energy_out

    @cached_property
    def exergy_balance(self):
        exergy_in = self.exergy_in
        exergy_out = self.exergy_out
        if exergy_in is None or exergy_out is None:
            return None
        return exergy_in - exergy_out
