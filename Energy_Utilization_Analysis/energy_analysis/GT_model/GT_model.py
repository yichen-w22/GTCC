import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass, field

from energy_analysis.working_fluid.gas import GasComposition, GasState

from energy_analysis.GT_model.components.chamber import Chamber, ChamberResult
from energy_analysis.GT_model.components.compressor import Compressor, CompressorResult
from energy_analysis.GT_model.config import GTModelConfig
from energy_analysis.GT_model.components.turbine import Turbine, TurbineResult


@dataclass
class GTModelResult:
    state_1: GasState
    state_2: GasState
    fuel: GasState
    state_3: GasState
    state_4: GasState
    compressor: CompressorResult
    chamber: ChamberResult
    turbine: TurbineResult
    net_power: float | None


@dataclass
class GTModel:
    config: GTModelConfig = field(default_factory=GTModelConfig)
    compressor: Compressor = field(default_factory=lambda: Compressor(name="compressor"))
    chamber: Chamber = field(default_factory=lambda: Chamber(name="chamber"))
    turbine: Turbine = field(default_factory=lambda: Turbine(name="turbine"))

    def solve(
        self,
        state_1: GasState,
        state_2: GasState,
        fuel: GasState,
        state_4: GasState,
        compressor_bleeding_mass_fraction: float | None = None,
        compressor_bleeding_pressure_fraction: float | None = None,
        compressor_bleeding_energy_fraction: float | None = None,
        total_pressure_recovery: float | None = None,
        combustion_efficiency: float | None = None,
    ) -> GTModelResult:
        
        if compressor_bleeding_mass_fraction is None:
            compressor_bleeding_mass_fraction = self.config.compressor_bleeding_mass_fraction
        if compressor_bleeding_pressure_fraction is None:
            compressor_bleeding_pressure_fraction = self.config.compressor_bleeding_pressure_fraction
        if compressor_bleeding_energy_fraction is None:
            compressor_bleeding_energy_fraction = self.config.compressor_bleeding_energy_fraction
        if total_pressure_recovery is None:
            total_pressure_recovery = self.config.total_pressure_recovery
        if combustion_efficiency is None:
            combustion_efficiency = self.config.combustion_efficiency

        compressor_result = self.compressor.solve(
            inlet_gas=state_1,
            outlet_gas=state_2,
            bleeding_mass_fraction=compressor_bleeding_mass_fraction,
            bleeding_pressure_fraction=compressor_bleeding_pressure_fraction,
            bleeding_energy_fraction=compressor_bleeding_energy_fraction,
        )
        chamber_result = self.chamber.solve(
            inlet_air=compressor_result.state_2,
            inlet_fuel=fuel,
            outlet_compositon=state_4.composition,
            total_pressure_recovery=total_pressure_recovery,
            combustion_efficiency=combustion_efficiency,
        )
        turbine_result = self.turbine.solve(
            state_3=chamber_result.state_3,
            state_4=state_4,
            bleeding=compressor_result.bleeding,
        )

        net_power = turbine_result.power - compressor_result.power

        return GTModelResult(
            state_1=state_1,
            state_2=compressor_result.state_2,
            fuel=fuel,
            state_3=chamber_result.state_3,
            state_4=turbine_result.state_4,
            compressor=compressor_result,
            chamber=chamber_result,
            turbine=turbine_result,
            net_power=net_power,
        )
