from dataclasses import dataclass

from .converter import EnergyConverter
from energy_analysis.working_fluid.steam_water import WaterSteamState


@dataclass
class ThrottleValveResult:
    inlet: WaterSteamState
    outlet: WaterSteamState
    pressure_drop: float | None
    enthalpy_change: float | None
    mass_balance: float | None
    energy_balance: float | None
    exergy_balance: float | None


@dataclass
class ThrottleValve(EnergyConverter):
    def pressure_drop(self) -> float | None:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        if inlet.P is None or outlet.P is None:
            return None
        return inlet.P - outlet.P

    def enthalpy_change(self) -> float | None:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        if inlet.h is None or outlet.h is None:
            return None
        return outlet.h - inlet.h

    def solve(self) -> ThrottleValveResult:
        return ThrottleValveResult(
            inlet=self.inlets[0],
            outlet=self.outlets[0],
            pressure_drop=self.pressure_drop(),
            enthalpy_change=self.enthalpy_change(),
            mass_balance=self.mass_balance(),
            energy_balance=self.energy_balance(),
            exergy_balance=self.exergy_balance(),
        )
