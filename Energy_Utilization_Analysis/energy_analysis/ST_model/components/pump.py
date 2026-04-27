from dataclasses import dataclass

from .converter import EnergyConverter
from energy_analysis.working_fluid.steam_water import WaterSteamState


@dataclass
class PumpResult:
    inlet: WaterSteamState
    outlet: WaterSteamState
    outlet_s: WaterSteamState
    isentropic_efficiency: float | None
    specific_work: float | None
    power_input: float | None
    mass_balance: float | None
    energy_balance: float | None
    exergy_balance: float | None


@dataclass
class Pump(EnergyConverter):
    def ideal_outlet_state(self) -> WaterSteamState:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        return WaterSteamState.from_Ps(
            P=outlet.P,
            s=inlet.s,
            m_dot=outlet.m_dot,
            name=f"{self.name}_outlet_s",
            ref=inlet.ref,
        )

    def specific_work(self) -> float | None:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        if inlet.h is None or outlet.h is None:
            return None
        return outlet.h - inlet.h

    def power_input(self) -> float | None:
        outlet = self.outlets[0]
        specific_work = self.specific_work()
        if outlet.m_dot is None or specific_work is None:
            return None
        return outlet.m_dot * specific_work

    def isentropic_efficiency(self) -> tuple[float | None, WaterSteamState]:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        outlet_s = self.ideal_outlet_state()
        if inlet.h is None or outlet.h is None or outlet_s.h is None:
            return None, outlet_s
        denominator = outlet.h - inlet.h
        if denominator == 0:
            return None, outlet_s
        return (outlet_s.h - inlet.h) / denominator, outlet_s

    def solve(self) -> PumpResult:
        efficiency, outlet_s = self.isentropic_efficiency()
        return PumpResult(
            inlet=self.inlets[0],
            outlet=self.outlets[0],
            outlet_s=outlet_s,
            isentropic_efficiency=efficiency,
            specific_work=self.specific_work(),
            power_input=self.power_input(),
            mass_balance=self.mass_balance(),
            energy_balance=self.energy_balance(),
            exergy_balance=self.exergy_balance(),
        )
