"""Steam-cycle component classes."""

from importlib import import_module

_EXPORTS = {
    "Condenser": ("energy_analysis.ST_model.components.condenser", "Condenser"),
    "CondenserResult": ("energy_analysis.ST_model.components.condenser", "CondenserResult"),
    "EnergyConverter": ("energy_analysis.ST_model.components.converter", "EnergyConverter"),
    "GasWaterHeatExchanger": ("energy_analysis.ST_model.components.heat_exchanger", "GasWaterHeatExchanger"),
    "GasWaterHeatExchangerResult": ("energy_analysis.ST_model.components.heat_exchanger", "GasWaterHeatExchangerResult"),
    "Mixer": ("energy_analysis.ST_model.components.mixer", "Mixer"),
    "MixerResult": ("energy_analysis.ST_model.components.mixer", "MixerResult"),
    "Pump": ("energy_analysis.ST_model.components.pump", "Pump"),
    "PumpResult": ("energy_analysis.ST_model.components.pump", "PumpResult"),
    "ThrottleValve": ("energy_analysis.ST_model.components.throttle_valve", "ThrottleValve"),
    "ThrottleValveResult": ("energy_analysis.ST_model.components.throttle_valve", "ThrottleValveResult"),
    "Turbine": ("energy_analysis.ST_model.components.turbine", "Turbine"),
    "TurbineResult": ("energy_analysis.ST_model.components.turbine", "TurbineResult"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
