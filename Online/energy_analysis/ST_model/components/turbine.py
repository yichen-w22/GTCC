from dataclasses import dataclass
from functools import cached_property

from .converter import EnergyConverter, EnergyConverterResult
from energy_analysis.working_fluid.steam_water import WaterSteamState


@dataclass
class TurbineResult(EnergyConverterResult):
    outlet_s: WaterSteamState | None = None
    isentropic_efficiency: float | None = None
    specific_work: float | None = None
    power_output: float | None = None
    mass_balance: float | None = None
    energy_balance: float | None = None
    exergy_balance: float | None = None


@dataclass
class Turbine(EnergyConverter):
    @cached_property
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
    
    @cached_property
    def specific_work(self) -> float | None:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        if inlet.h is None or outlet.h is None:
            return None
        return inlet.h - outlet.h
    
    @cached_property
    def power_output(self) -> float | None:
        outlet = self.outlets[0]
        specific_work = self.specific_work
        if outlet.m_dot is None or specific_work is None:
            return None
        return outlet.m_dot * specific_work

    @cached_property
    def isentropic_efficiency(self) -> tuple[float | None, WaterSteamState]:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        outlet_s = self.ideal_outlet_state
        if inlet.h is None or outlet.h is None or outlet_s.h is None:
            return None, outlet_s
        denominator = inlet.h - outlet_s.h
        if denominator == 0:
            return None, outlet_s
        return (inlet.h - outlet.h) / denominator, outlet_s

    @cached_property
    def exergy_efficiency(self) -> tuple[float | None, WaterSteamState]:
        inlet = self.inlets[0]
        outlet = self.outlets[0]
        return (inlet.h - outlet.h) / (inlet.exergy - outlet.exergy)