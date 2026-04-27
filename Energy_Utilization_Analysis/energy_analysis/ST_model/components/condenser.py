from dataclasses import dataclass

from .converter import EnergyConverter


@dataclass
class CondenserResult:
    inlet: object
    outlet: object
    heat_rejection: float | None
    mass_balance: float | None
    energy_balance: float | None
    exergy_balance: float | None


@dataclass
class Condenser(EnergyConverter):
    def heat_rejection(self) -> float | None:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        if inlet.energy_flow is None or outlet.energy_flow is None:
            return None
        return inlet.energy_flow - outlet.energy_flow

    def solve(self) -> CondenserResult:
        return CondenserResult(
            inlet=self.inlets[0],
            outlet=self.outlets[0],
            heat_rejection=self.heat_rejection(),
            mass_balance=self.mass_balance(),
            energy_balance=self.energy_balance(),
            exergy_balance=self.exergy_balance(),
        )
