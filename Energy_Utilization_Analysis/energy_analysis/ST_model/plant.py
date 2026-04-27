from pathlib import Path

import pandas as pd

from energy_analysis.working_fluid.streams import build_streams_from_row, build_gases_from_row
from energy_analysis.ST_model.components.condenser import Condenser
from energy_analysis.ST_model.components.mixer import Mixer
from energy_analysis.ST_model.components.pump import Pump
from energy_analysis.ST_model.components.turbine import Turbine
from energy_analysis.ST_model.components.heat_exchanger import GasWaterHeatExchanger
from energy_analysis.ST_model.components.throttle_valve import ThrottleValve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data_precessing" / "continuous_data_10min.csv"

def build_plant_from_streams(streams: dict, gases: dict):
    plant = {}

    # 余热锅炉
    plant["hrsg1"] = GasWaterHeatExchanger(name="hrsg1")
    plant["hrsg1"].add_inlet(gases["1号余热锅炉入口烟气"])
    plant["hrsg1"].add_outlet(gases["1号余热锅炉出口烟气"])
    plant["hrsg1"].add_inlet(streams["1号炉高压省煤器入口"])
    plant["hrsg1"].add_outlet(streams["1号炉高压过热器出口"])
    plant["hrsg1"].add_inlet(streams["1号炉高压减温器后"])
    plant["hrsg1"].add_outlet(streams["1号炉高压主蒸汽"])
    plant["hrsg1"].add_inlet(streams["1号炉中压省煤器入口"])
    plant["hrsg1"].add_outlet(streams["1号炉中压过热器出口"])
    plant["hrsg1"].add_inlet(streams["1号炉冷再热混合后"])
    plant["hrsg1"].add_outlet(streams["1号炉再热器1出口"])
    plant["hrsg1"].add_inlet(streams["1号炉再热减温器出口"])
    plant["hrsg1"].add_outlet(streams["1号炉热再热出口"])
    plant["hrsg1"].add_inlet(streams["1号炉低压省煤器入口"])
    plant["hrsg1"].add_outlet(streams["1号炉低压省煤器出口"])
    plant["hrsg1"].add_inlet(streams["1号炉低压汽包给水调节阀出口"])
    plant["hrsg1"].add_outlet(streams["1号炉低压汽包"])
    plant["hrsg1"].add_inlet(streams["1号炉低压汽包至低压主蒸汽"])
    plant["hrsg1"].add_outlet(streams["1号炉低压主蒸汽"])

    plant["hrsg2"] = GasWaterHeatExchanger(name="hrsg2") #！
    plant["hrsg2"].add_inlet(gases["2号余热锅炉入口烟气"])
    plant["hrsg2"].add_outlet(gases["2号余热锅炉出口烟气"])
    plant["hrsg2"].add_inlet(streams["2号炉高压省煤器入口"])
    plant["hrsg2"].add_outlet(streams["2号炉高压过热器出口"])
    plant["hrsg2"].add_inlet(streams["2号炉高压减温器后"])
    plant["hrsg2"].add_outlet(streams["2号炉高压主蒸汽"])
    plant["hrsg2"].add_inlet(streams["2号炉中压省煤器入口"])
    plant["hrsg2"].add_outlet(streams["2号炉中压过热器出口"])
    plant["hrsg2"].add_inlet(streams["2号炉冷再热混合后"])
    plant["hrsg2"].add_outlet(streams["2号炉再热器1出口"])
    plant["hrsg2"].add_inlet(streams["2号炉再热减温器出口"])
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

    plant["condenser"] = Condenser(name="condenser")
    plant["condenser"].add_inlet(streams["低压缸出口"])
    plant["condenser"].add_outlet(streams["凝汽器出口"])

    # 凝结水泵
    plant["condensate_pump"] = Pump(name="condensate_pump")
    plant["condensate_pump"].add_inlet(streams["凝汽器出口"])
    plant["condensate_pump"].add_outlet(streams["凝结水泵出口"])

    # 给水泵
    plant["ip_1_pump"] = Pump(name="ip_1_pump")
    plant["ip_1_pump"].add_inlet(streams["1号炉低压汽包中压给水泵前"])
    plant["ip_1_pump"].add_outlet(streams["1号炉低压汽包中压给水泵后"])

    plant["ip_2_pump"] = Pump(name="ip_2_pump")
    plant["ip_2_pump"].add_inlet(streams["2号炉低压汽包中压给水泵前"])
    plant["ip_2_pump"].add_outlet(streams["2号炉低压汽包中压给水泵后"])

    plant["hp_1_pump"] = Pump(name="hp_1_pump")
    plant["hp_1_pump"].add_inlet(streams["1号炉低压汽包高压给水泵前"])
    plant["hp_1_pump"].add_outlet(streams["1号炉低压汽包高压给水泵后"])

    plant["hp_2_pump"] = Pump(name="hp_2_pump")
    plant["hp_2_pump"].add_inlet(streams["2号炉低压汽包高压给水泵前"])
    plant["hp_2_pump"].add_outlet(streams["2号炉低压汽包高压给水泵后"])

    # 节流阀
    plant["throttle_valve_1"] = ThrottleValve(name="throttle_valve_1")
    plant["throttle_valve_1"].add_inlet(streams["1号炉低压省煤器出口"])
    plant["throttle_valve_1"].add_outlet(streams["1号炉低压汽包给水调节阀出口"])

    plant["throttle_valve_2"] = ThrottleValve(name="throttle_valve_2")
    plant["throttle_valve_2"].add_inlet(streams["2号炉低压省煤器出口"])
    plant["throttle_valve_2"].add_outlet(streams["2号炉低压汽包给水调节阀出口"])

    # 减温器
    plant["hp_1_cooler"] = Mixer(name="hp_1_cooler")
    plant["hp_1_cooler"].add_inlet(streams["1号炉高压过热器出口"])
    plant["hp_1_cooler"].add_inlet(streams["1号炉高压减温水"])
    plant["hp_1_cooler"].add_outlet(streams["1号炉高压减温器后"])

    plant["hp_2_cooler"] = Mixer(name="hp_2_cooler")
    plant["hp_2_cooler"].add_inlet(streams["2号炉高压过热器出口"])
    plant["hp_2_cooler"].add_inlet(streams["2号炉高压减温水"])
    plant["hp_2_cooler"].add_outlet(streams["2号炉高压减温器后"])

    plant["rh_1_cooler"] = Mixer(name="rh_1_cooler")
    plant["rh_1_cooler"].add_inlet(streams["1号炉再热器1出口"])
    plant["rh_1_cooler"].add_inlet(streams["1号炉中压减温水"])
    plant["rh_1_cooler"].add_outlet(streams["1号炉再热减温器出口"])

    plant["rh_2_cooler"] = Mixer(name="rh_2_cooler")
    plant["rh_2_cooler"].add_inlet(streams["2号炉再热器1出口"])
    plant["rh_2_cooler"].add_inlet(streams["2号炉中压减温水"])
    plant["rh_2_cooler"].add_outlet(streams["2号炉再热减温器出口"])

    # 再热混合
    plant["ip_crh_1"] = Mixer(name="ip_crh_1")  # 中压主蒸汽与冷再热蒸汽混合
    plant["ip_crh_1"].add_inlet(streams["1号炉中压过热器出口"])
    plant["ip_crh_1"].add_inlet(streams["1号炉高压缸排汽"])
    plant["ip_crh_1"].add_outlet(streams["1号炉冷再热混合后"])

    plant["ip_crh_2"] = Mixer(name="ip_crh_2")
    plant["ip_crh_2"].add_inlet(streams["2号炉中压过热器出口"])
    plant["ip_crh_2"].add_inlet(streams["2号炉高压缸排汽"])
    plant["ip_crh_2"].add_outlet(streams["2号炉冷再热混合后"])

    # 出口主蒸汽合并
    plant["hp"] = Mixer(name="hp")
    plant["hp"].add_inlet(streams["1号炉高压主蒸汽"])
    plant["hp"].add_inlet(streams["2号炉高压主蒸汽"])
    plant["hp"].add_outlet(streams["高压缸入口"])

    plant["ip"] = Mixer(name="ip")
    plant["ip"].add_inlet(streams["1号炉热再热出口"])
    plant["ip"].add_inlet(streams["2号炉热再热出口"])
    plant["ip"].add_outlet(streams["中压缸入口"])

    plant["lp"] = Mixer(name="lp")
    plant["lp"].add_inlet(streams["1号炉低压主蒸汽"])
    plant["lp"].add_inlet(streams["2号炉低压主蒸汽"])
    plant["lp"].add_inlet(streams["中压缸出口"])
    plant["lp"].add_outlet(streams["低压缸入口"])

    return plant


def build_plant(idx: int = 100, data_path: str | Path = DEFAULT_DATA_PATH):
    df = pd.read_csv(data_path)
    streams = build_streams_from_row(df, idx)
    gases = build_gases_from_row(df, idx)
    return build_plant_from_streams(streams, gases)
