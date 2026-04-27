from dataclasses import dataclass

from .converter import EnergyConverter
from energy_analysis.working_fluid.steam_water import WaterSteamState


@dataclass
class MixerResult:
    inlets: list[WaterSteamState]
    outlet: WaterSteamState
    mass_balance: float | None
    energy_balance: float | None
    exergy_balance: float | None


@dataclass
class Mixer(EnergyConverter):
    def solve(self) -> MixerResult:
        return MixerResult(
            inlets=list(self.inlets),
            outlet=self.outlets[0],
            mass_balance=self.mass_balance(),
            energy_balance=self.energy_balance(),
            exergy_balance=self.exergy_balance(),
        )
