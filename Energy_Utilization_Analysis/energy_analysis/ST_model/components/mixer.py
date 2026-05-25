from dataclasses import dataclass

from .converter import EnergyConverter, EnergyConverterResult


@dataclass
class MixerResult(EnergyConverterResult):
    mass_balance: float | None = None
    energy_balance: float | None = None
    exergy_balance: float | None = None


@dataclass
class Mixer(EnergyConverter):
    def solve(self) -> MixerResult:
        return MixerResult(
            inlets=list(self.inlets),
            outlets=list(self.outlets),
            mass_balance=self.mass_balance,
            energy_balance=self.energy_balance,
            exergy_balance=self.exergy_balance,
        )
