from dataclasses import dataclass
from functools import cached_property

from .converter import EnergyConverter, EnergyConverterResult


@dataclass
class CondenserResult(EnergyConverterResult):
    heat_rejection: float | None = None
    mass_balance: float | None = None
    energy_balance: float | None = None
    exergy_balance: float | None = None


@dataclass
class Condenser(EnergyConverter):
    @cached_property
    def heat_rejection(self) -> float | None:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        if inlet.energy_flow is None or outlet.energy_flow is None:
            return None
        return inlet.energy_flow - outlet.energy_flow

    def solve(self) -> CondenserResult:
        return CondenserResult(
            inlets=list(self.inlets),
            outlets=list(self.outlets),
            heat_rejection=self.heat_rejection,
            mass_balance=self.mass_balance,
            energy_balance=self.energy_balance,
            exergy_balance=self.exergy_balance,
        )
