import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass
from typing import Any

import pandas as pd

from energy_analysis.ST_model.plant import DEFAULT_DATA_PATH, build_plant_from_streams
from energy_analysis.working_fluid.fluid import FlowState
from energy_analysis.working_fluid.streams import build_gases_from_row, build_streams_from_row


@dataclass
class STModelResult:
    streams: dict[str, FlowState]
    gases: dict[str, FlowState]
    plant: dict[str, Any]
    component_results: dict[str, Any]
    stream_results: dict[str, dict[str, Any]]
    gas_results: dict[str, dict[str, Any]]
    total_turbine_power: float | None
    total_pump_power: float | None
    net_power: float | None
    max_mass_balance_error: float | None
    max_energy_balance_error: float | None
    max_exergy_balance_error: float | None


@dataclass
class STModel:
    def _flatten_stream(self, stream: FlowState) -> dict[str, Any]:
        result = {
            "name": stream.name,
            "T": stream.T,
            "P": stream.P,
            "m_dot": stream.m_dot,
            "h": stream.h,
            "s": stream.s,
            "energy_flow": stream.energy_flow,
            "exergy": stream.exergy,
            "exergy_flow": stream.exergy_flow,
        }
        if hasattr(stream, "x"):
            result["x"] = getattr(stream, "x")
        if hasattr(stream, "composition"):
            composition = getattr(stream, "composition")
            result["composition"] = composition.as_dict() if composition is not None else None
        return result

    @staticmethod
    def _max_abs(values: list[float | None]) -> float | None:
        defined_values = [abs(value) for value in values if value is not None]
        if not defined_values:
            return None
        return max(defined_values)

    @staticmethod
    def _sum_defined(values: list[float | None]) -> float | None:
        if any(value is None for value in values):
            return None
        return sum(values)

    def solve(
        self,
        idx: int = 100,
        data_path: str | Path = DEFAULT_DATA_PATH,
    ) -> STModelResult:
        df = pd.read_csv(data_path)
        streams = build_streams_from_row(df, idx)
        gases = build_gases_from_row(df, idx)
        plant = build_plant_from_streams(streams, gases)

        component_results = {name: component.solve() for name, component in plant.items()}
        stream_results = {name: self._flatten_stream(stream) for name, stream in streams.items()}
        gas_results = {name: self._flatten_stream(gas) for name, gas in gases.items()}

        turbine_names = ["hp_turbine", "ip_turbine", "lp_turbine"]
        pump_names = ["condensate_pump", "ip_1_pump", "ip_2_pump", "hp_1_pump", "hp_2_pump"]

        total_turbine_power = self._sum_defined(
            [component_results[name].power_output for name in turbine_names]
        )
        total_pump_power = self._sum_defined(
            [component_results[name].power_input for name in pump_names]
        )
        net_power = None
        if total_turbine_power is not None and total_pump_power is not None:
            net_power = total_turbine_power - total_pump_power

        mass_balance_errors = [getattr(result, "mass_balance", None) for result in component_results.values()]
        energy_balance_errors = [getattr(result, "energy_balance", None) for result in component_results.values()]
        exergy_balance_errors = [getattr(result, "exergy_balance", None) for result in component_results.values()]

        return STModelResult(
            streams=streams,
            gases=gases,
            plant=plant,
            component_results=component_results,
            stream_results=stream_results,
            gas_results=gas_results,
            total_turbine_power=total_turbine_power,
            total_pump_power=total_pump_power,
            net_power=net_power,
            max_mass_balance_error=self._max_abs(mass_balance_errors),
            max_energy_balance_error=self._max_abs(energy_balance_errors),
            max_exergy_balance_error=self._max_abs(exergy_balance_errors),
        )
