"""Public API for the energy analysis package."""

from importlib import import_module

_EXPORTS = {
    "GTModel": ("energy_analysis.GT_model", "GTModel"),
    "GTModelConfig": ("energy_analysis.GT_model", "GTModelConfig"),
    "Chamber": ("energy_analysis.GT_model", "Chamber"),
    "Compressor": ("energy_analysis.GT_model", "Compressor"),
    "GTTurbine": ("energy_analysis.GT_model", "GTTurbine"),
    "STModel": ("energy_analysis.ST_model", "STModel"),
    "STModelResult": ("energy_analysis.ST_model", "STModelResult"),
    "build_plant": ("energy_analysis.ST_model", "build_plant"),
    "build_plant_from_streams": ("energy_analysis.ST_model", "build_plant_from_streams"),
    "Condenser": ("energy_analysis.ST_model", "Condenser"),
    "CondenserResult": ("energy_analysis.ST_model", "CondenserResult"),
    "EnergyConverter": ("energy_analysis.ST_model", "EnergyConverter"),
    "GasWaterHeatExchanger": ("energy_analysis.ST_model", "GasWaterHeatExchanger"),
    "GasWaterHeatExchangerResult": ("energy_analysis.ST_model", "GasWaterHeatExchangerResult"),
    "Mixer": ("energy_analysis.ST_model", "Mixer"),
    "MixerResult": ("energy_analysis.ST_model", "MixerResult"),
    "Pump": ("energy_analysis.ST_model", "Pump"),
    "PumpResult": ("energy_analysis.ST_model", "PumpResult"),
    "STTurbine": ("energy_analysis.ST_model", "STTurbine"),
    "STTurbineResult": ("energy_analysis.ST_model", "STTurbineResult"),
    "ThrottleValve": ("energy_analysis.ST_model", "ThrottleValve"),
    "ThrottleValveResult": ("energy_analysis.ST_model", "ThrottleValveResult"),
    "FlowState": ("energy_analysis.working_fluid", "FlowState"),
    "GasComposition": ("energy_analysis.working_fluid", "GasComposition"),
    "GasReferenceEnv": ("energy_analysis.working_fluid", "GasReferenceEnv"),
    "GasState": ("energy_analysis.working_fluid", "GasState"),
    "ReferenceEnv": ("energy_analysis.working_fluid", "ReferenceEnv"),
    "WaterSteamState": ("energy_analysis.working_fluid", "WaterSteamState"),
    "build_air_composition": ("energy_analysis.working_fluid", "build_air_composition"),
    "build_flue_gas_composition": ("energy_analysis.working_fluid", "build_flue_gas_composition"),
    "build_fuel_composition_from_row": ("energy_analysis.working_fluid", "build_fuel_composition_from_row"),
    "build_gases_from_row": ("energy_analysis.working_fluid", "build_gases_from_row"),
    "build_streams_from_row": ("energy_analysis.working_fluid", "build_streams_from_row"),
    "calc_fuel_lhv": ("energy_analysis.working_fluid", "calc_fuel_lhv"),
    "calc_gas_density": ("energy_analysis.working_fluid", "calc_gas_density"),
    "create_gas_reference_env": ("energy_analysis.working_fluid", "create_gas_reference_env"),
    "create_reference_env": ("energy_analysis.working_fluid", "create_reference_env"),
    "create_water_reference_env": ("energy_analysis.working_fluid", "create_water_reference_env"),
    "mixture_h_s_cp": ("energy_analysis.working_fluid", "mixture_h_s_cp"),
    "relative_humidity_to_mole_fraction": ("energy_analysis.working_fluid", "relative_humidity_to_mole_fraction"),
    "solve_temperature_from_property": ("energy_analysis.working_fluid", "solve_temperature_from_property"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
