from dataclasses import dataclass
from functools import cached_property

from .converter import EnergyConverter, EnergyConverterResult


@dataclass
class ThrottleValveResult(EnergyConverterResult):
    pressure_drop: float | None = None
    enthalpy_change: float | None = None
    mass_balance: float | None = None
    energy_balance: float | None = None
    exergy_balance: float | None = None


@dataclass
class ThrottleValve(EnergyConverter):
    @cached_property
    def pressure_drop(self) -> float | None:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        if inlet.P is None or outlet.P is None:
            return None
        return inlet.P - outlet.P

    @cached_property
    def enthalpy_change(self) -> float | None:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        if inlet.h is None or outlet.h is None:
            return None
        return outlet.h - inlet.h

    def solve(self) -> ThrottleValveResult:
        return ThrottleValveResult(
            inlets=list(self.inlets),
            outlets=list(self.outlets),
            pressure_drop=self.pressure_drop,
            enthalpy_change=self.enthalpy_change,
            mass_balance=self.mass_balance,
            energy_balance=self.energy_balance,
            exergy_balance=self.exergy_balance,
        )
