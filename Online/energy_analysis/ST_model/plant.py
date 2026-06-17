from pathlib import Path

import pandas as pd

from energy_analysis.working_fluid.streams import build_streams_from_row, build_gases_from_row
from energy_analysis.ST_model.components.turbine import Turbine
from energy_analysis.ST_model.components.heat_exchanger import GasWaterHeatExchanger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data_precessing" / "continuous_data_10min.csv"

def build_plant_from_streams(streams: dict, gases: dict):
    plant = {}

    # 余热锅炉
    plant["hrsg1"] = GasWaterHeatExchanger(name="hrsg1")
    plant["hrsg1"].add_inlet(gases["1号余热锅炉入口烟气"])
    plant["hrsg1"].add_outlet(gases["1号余热锅炉出口烟气"])
    plant["hrsg1"].add_inlet(streams["1号炉低压汽包高压给水泵后"])
    plant["hrsg1"].add_outlet(streams["1号炉高压主蒸汽"])
    plant["hrsg1"].add_inlet(streams["1号炉低压汽包中压给水泵后"])
    plant["hrsg1"].add_inlet(streams["1号炉高压缸排汽"])
    plant["hrsg1"].add_outlet(streams["1号炉热再热出口"])
    plant["hrsg1"].add_inlet(streams["1号炉低压省煤器入口"])
    plant["hrsg1"].add_outlet(streams["1号炉低压省煤器出口"])
    plant["hrsg1"].add_inlet(streams["1号炉低压汽包给水调节阀出口"])
    plant["hrsg1"].add_outlet(streams["1号炉低压汽包"])
    plant["hrsg1"].add_inlet(streams["1号炉低压汽包至低压主蒸汽"])
    plant["hrsg1"].add_outlet(streams["1号炉低压主蒸汽"])

    plant["hrsg2"] = GasWaterHeatExchanger(name="hrsg2")
    plant["hrsg2"].add_inlet(gases["2号余热锅炉入口烟气"])
    plant["hrsg2"].add_outlet(gases["2号余热锅炉出口烟气"])
    plant["hrsg2"].add_inlet(streams["2号炉低压汽包高压给水泵后"])
    plant["hrsg2"].add_outlet(streams["2号炉高压主蒸汽"])
    plant["hrsg2"].add_inlet(streams["2号炉低压汽包中压给水泵后"])
    plant["hrsg2"].add_inlet(streams["2号炉高压缸排汽"])
    plant["hrsg2"].add_outlet(streams["2号炉热再热出口"])
    plant["hrsg2"].add_inlet(streams["2号炉低压省煤器入口"])
    plant["hrsg2"].add_outlet(streams["2号炉低压省煤器出口"])
    plant["hrsg2"].add_inlet(streams["2号炉低压汽包给水调节阀出口"])
    plant["hrsg2"].add_outlet(streams["2号炉低压汽包"])
    plant["hrsg2"].add_inlet(streams["2号炉低压汽包至低压主蒸汽"])
    plant["hrsg2"].add_outlet(streams["2号炉低压主蒸汽"])

    # 汽机
    plant["hp_turbine"] = Turbine(name="hp_turbine")
    plant["hp_turbine"].add_inlet(streams["高压缸入口"])
    plant["hp_turbine"].add_outlet(streams["高压缸出口"])

    plant["ip_turbine"] = Turbine(name="ip_turbine")
    plant["ip_turbine"].add_inlet(streams["中压缸入口"])
    plant["ip_turbine"].add_outlet(streams["中压缸出口"])

    plant["lp_turbine"] = Turbine(name="lp_turbine")
    plant["lp_turbine"].add_inlet(streams["低压缸入口"])
    plant["lp_turbine"].add_outlet(streams["低压缸出口"])

    return plant


def build_plant(df: pd.DataFrame = None, idx: int = 100):
    streams = build_streams_from_row(df, idx)
    gases = build_gases_from_row(df, idx)
    return build_plant_from_streams(streams, gases)
