from dataclasses import dataclass
from functools import cached_property

from .converter import EnergyConverter, EnergyConverterResult


@dataclass
class GasWaterHeatExchanger(EnergyConverter):
    @cached_property
    def gas_energy_release(self):
        return self.inlets[0].energy_flow - self.outlets[0].energy_flow

    @cached_property
    def water_energy_absorption(self):
        q = 0.0
        for s_in in self.inlets[1:]:
            q -= s_in.energy_flow 
        for s_out in self.outlets[1:]:
            q += s_out.energy_flow
        return q

    @cached_property
    def energy_loss(self):
        gas_energy_release = self.gas_energy_release
        water_energy_absorption = self.water_energy_absorption
        return gas_energy_release - water_energy_absorption

    @cached_property
    def energy_balance_ratio(self):
        gas_energy_release = self.gas_energy_release
        water_energy_absorption = self.water_energy_absorption
        return water_energy_absorption / gas_energy_release

    @cached_property
    def energy_effectiveness(self): 
        return self.gas_energy_release / self.inlets[0].energy_flow

    @cached_property
    def gas_exergy_release(self):
        return self.inlets[0].exergy_flow - self.outlets[0].exergy_flow

    @cached_property
    def water_exergy_absorption(self):
        ex = 0.0
        for s_in, s_out in zip(self.inlets[1:], self.outlets[1:]):
            ex += s_out.exergy_flow - s_in.exergy_flow
        return ex

    @cached_property
    def exergy_balance_ratio(self):
        gas_exergy_release = self.gas_exergy_release
        water_exergy_absorption = self.water_exergy_absorption
        return water_exergy_absorption / gas_exergy_release

    @cached_property
    def exergy_effectiveness(self): 
        return self.gas_exergy_release / self.inlets[0].exergy_flow
