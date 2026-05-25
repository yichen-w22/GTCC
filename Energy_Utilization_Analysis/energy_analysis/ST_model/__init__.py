"""Steam-turbine and plant model public API."""

from importlib import import_module

_EXPORTS = {
    "STModel": ("energy_analysis.ST_model.ST_model", "STModel"),
    "STModelResult": ("energy_analysis.ST_model.ST_model", "STModelResult"),
    "build_plant": ("energy_analysis.ST_model.plant", "build_plant"),
    "build_plant_from_streams": ("energy_analysis.ST_model.plant", "build_plant_from_streams"),
    "Condenser": ("energy_analysis.ST_model.components", "Condenser"),
    "CondenserResult": ("energy_analysis.ST_model.components", "CondenserResult"),
    "EnergyConverter": ("energy_analysis.ST_model.components", "EnergyConverter"),
    "GasWaterHeatExchanger": ("energy_analysis.ST_model.components", "GasWaterHeatExchanger"),
    "GasWaterHeatExchangerResult": ("energy_analysis.ST_model.components", "GasWaterHeatExchangerResult"),
    "Mixer": ("energy_analysis.ST_model.components", "Mixer"),
    "MixerResult": ("energy_analysis.ST_model.components", "MixerResult"),
    "Pump": ("energy_analysis.ST_model.components", "Pump"),
    "PumpResult": ("energy_analysis.ST_model.components", "PumpResult"),
    "STTurbine": ("energy_analysis.ST_model.components", "Turbine"),
    "STTurbineResult": ("energy_analysis.ST_model.components", "TurbineResult"),
    "ThrottleValve": ("energy_analysis.ST_model.components", "ThrottleValve"),
    "ThrottleValveResult": ("energy_analysis.ST_model.components", "ThrottleValveResult"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
