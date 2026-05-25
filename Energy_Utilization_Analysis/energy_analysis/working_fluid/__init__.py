"""Working-fluid states and helper builders."""

from importlib import import_module

_EXPORTS = {
    "FlowState": ("energy_analysis.working_fluid.fluid", "FlowState"),
    "GasComposition": ("energy_analysis.working_fluid.gas", "GasComposition"),
    "GasReferenceEnv": ("energy_analysis.working_fluid.gas", "GasReferenceEnv"),
    "GasState": ("energy_analysis.working_fluid.gas", "GasState"),
    "build_air_composition": ("energy_analysis.working_fluid.gas", "build_air_composition"),
    "build_flue_gas_composition": ("energy_analysis.working_fluid.gas", "build_flue_gas_composition"),
    "calc_fuel_lhv": ("energy_analysis.working_fluid.gas", "calc_fuel_lhv"),
    "calc_gas_density": ("energy_analysis.working_fluid.gas", "calc_gas_density"),
    "create_gas_reference_env": ("energy_analysis.working_fluid.gas", "create_gas_reference_env"),
    "mixture_h_s_cp": ("energy_analysis.working_fluid.gas", "mixture_h_s_cp"),
    "relative_humidity_to_mole_fraction": ("energy_analysis.working_fluid.gas", "relative_humidity_to_mole_fraction"),
    "solve_temperature_from_property": ("energy_analysis.working_fluid.gas", "solve_temperature_from_property"),
    "ReferenceEnv": ("energy_analysis.working_fluid.steam_water", "ReferenceEnv"),
    "WaterSteamState": ("energy_analysis.working_fluid.steam_water", "WaterSteamState"),
    "create_reference_env": ("energy_analysis.working_fluid.steam_water", "create_reference_env"),
    "create_water_reference_env": ("energy_analysis.working_fluid.steam_water", "create_water_reference_env"),
    "build_fuel_composition_from_row": ("energy_analysis.working_fluid.streams", "build_fuel_composition_from_row"),
    "build_gases_from_row": ("energy_analysis.working_fluid.streams", "build_gases_from_row"),
    "build_streams_from_row": ("energy_analysis.working_fluid.streams", "build_streams_from_row"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
