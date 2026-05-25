import sys
from functools import cached_property
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass, field

from energy_analysis.GT_model.components.chamber import Chamber
from energy_analysis.GT_model.components.compressor import Compressor
from energy_analysis.GT_model.components.turbine import Turbine
from energy_analysis.GT_model.config import GTModelConfig
from energy_analysis.working_fluid.gas import GasState
from energy_analysis.working_fluid.streams import build_gases_from_row


@dataclass
class GTModel:
    state_1: GasState | None = None
    compressor_outlet: GasState | None = None
    fuel: GasState | None = None
    state_4: GasState | None = None
    config: GTModelConfig = field(default_factory=GTModelConfig)

    @cached_property
    def compressor(self) -> Compressor:
        return Compressor(
            name="compressor",
            inlet_gas=self.state_1,
            outlet_gas=self.compressor_outlet,
            bleeding_mass_fraction=self.config.compressor_bleeding_mass_fraction,
            bleeding_pressure_fraction=self.config.compressor_bleeding_pressure_fraction,
            bleeding_energy_fraction=self.config.compressor_bleeding_energy_fraction,
        )

    @cached_property
    def chamber(self) -> Chamber:
        return Chamber(
            name="chamber",
            inlet_air=self.compressor.state_2,
            inlet_fuel=self.fuel,
            outlet_composition=self.state_4.composition,
            total_pressure_recovery=self.config.total_pressure_recovery,
            combustion_efficiency=self.config.combustion_efficiency,
            bleeding=self.compressor.bleeding,
        )

    @cached_property
    def turbine(self) -> Turbine:
        return Turbine(
            name="turbine",
            state_3_c=self.chamber.state_3_c,
            state_4=self.state_4
        )

    @cached_property
    def state_2(self) -> GasState:
        return self.compressor.state_2

    @cached_property
    def state_3(self) -> GasState:
        return self.chamber.state_3
    
    @cached_property
    def state_3_c(self) -> GasState:
        return self.chamber.state_3_c

    @cached_property
    def net_power(self) -> float | None:
        if self.turbine.power is None or self.compressor.power is None:
            return None
        return self.turbine.power - self.compressor.power

    @cached_property
    def fuel_energy(self) -> float:
        return self.chamber.fuel_lhv * self.fuel.m_dot

    @cached_property
    def thermal_efficiency(self) -> float | None:
        if self.fuel_energy == 0:
            return None
        return self.net_power / self.fuel_energy

    @cached_property
    def generation_power_share(self) -> float | None:
        if self.turbine.power in (None, 0) or self.net_power is None:
            return None
        return self.net_power / self.turbine.power

    @cached_property
    def compressor_power_share(self) -> float | None:
        if self.turbine.power in (None, 0) or self.compressor.power is None:
            return None
        return self.compressor.power / self.turbine.power

    @cached_property
    def exhaust_energy_share(self) -> float | None:
        if self.turbine.power in (None, 0) or self.state_4.energy_flow is None:
            return None
        return self.state_4.energy_flow / self.turbine.power


def build_gt(df, idx, unit):
    gases = build_gases_from_row(df, idx)
    if unit == 1:
        unit_name = "1号"
    elif unit == 2:
        unit_name = "2号"
    else:
        raise ValueError("unit must be 1 or 2")

    return GTModel(
        state_1=gases[f"{unit_name}燃机入口空气"],
        compressor_outlet=gases[f"{unit_name}燃机压气机出口"],
        fuel=gases[f"{unit_name}炉燃料"],
        state_4=gases[f"{unit_name}余热锅炉入口烟气"],
    )
