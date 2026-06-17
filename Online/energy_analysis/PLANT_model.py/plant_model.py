import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.GT_model.GT_model import build_gt
from energy_analysis.ST_model.plant import build_plant
from energy_analysis.working_fluid.streams import build_streams_from_row
from data_precessing.online_input import DEFAULT_INPUT_PATH, build_online_dataframe_from_file

DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
OUTPUT_DIR = PROJECT_ROOT / "temp"
OUTPUT_PATH = OUTPUT_DIR / "plant_metrics2.csv"
JSON_OUTPUT_PATH = PROJECT_ROOT / "output" / "output.json"
GT_POWER_STOP_THRESHOLD = 1.0e6
GT_FUEL_FLOW_STOP_THRESHOLD = 5.0
GT_FLUE_GAS_FLOW_STOP_THRESHOLD_LOWER = 100.0
GT_FLUE_GAS_FLOW_STOP_THRESHOLD_UPPER = 1000.0
LP_CUT_PRESSURE_THRESHOLD = 0.5 * 101325.0
PERFORMANCE_EVENT_KEY = "JYRD.Performance.Computed"


def default_output_path_for_input(input_path):
    input_path = Path(input_path)
    return JSON_OUTPUT_PATH.parent / f"{input_path.stem}_outcome{input_path.suffix or '.json'}"


def identity(value):
    return value


def ratio_to_percent(value):
    if is_missing(value):
        return None
    return value * 100.0


def watt_to_mw(value):
    if is_missing(value):
        return None
    return value / 1.0e6


def first_available(metrics, keys):
    for key in keys:
        value = metrics.get(key)
        if not is_missing(value):
            return value
    return None


# 数据库变量名 -> (内部计算变量名/候选变量名, 单位换算函数, 真实含义)
# None 表示当前模型尚不能计算该变量，输出 JSON 时按 null 处理。
OUTPUT_MAPPING = {
    "SKY.GTCC.NLXL_100": ("PLANT.联合循环能量效率", ratio_to_percent, "联合循环能量效率"),
    "SKY.GTCC.FDXL_100": ("PLANT.联合循环发电效率", ratio_to_percent, "联合循环发电效率"),
    "SKY.GTCC.YXL_100": ("PLANT.联合循环㶲效率", ratio_to_percent, "联合循环㶲效率"),
    "SKY.GTCC.RH_MW": ("PLANT.热耗", watt_to_mw, "联合循环燃料热输入/热耗"),
    "SKY.GT1.RXL_100": ("GT1.燃机热效率", ratio_to_percent, "#1燃机热效率"),
    "SKY.GT1.CMP.DSXL_100": ("GT1.压气机等熵效率", ratio_to_percent, "#1燃机压气机等熵效率"),
    "SKY.GT1.CMP.HG_MW": ("GT1.压气机耗功", watt_to_mw, "#1燃机压气机耗功"),
    "SKY.GT1.COMB.T3_K": ("GT1.燃烧室温度", identity, "#1燃机燃烧室燃烧温度"),
    "SKY.GT1.RH_MW": ("GT1.燃料能量", watt_to_mw, "#1燃机燃料热输入/热耗"),
    "SKY.GT1.TURB.DSXL_100": ("GT1.透平等熵效率", ratio_to_percent, "#1燃机透平等熵效率"),
    "SKY.GT1.TURB.YXL_100": ("GT1.透平㶲效率", ratio_to_percent, "#1燃机透平㶲效率"),
    "SKY.GT1.TURB.ZG_MW": ("GT1.透平输出功", watt_to_mw, "#1燃机透平做功"),
    "SKY.GT1.CMP.DSSS_MW": ("GT1.压气机等熵损失", watt_to_mw, "#1燃机压气机等熵损失"),
    "SKY.GT1.TURB.DSSS_MW": ("GT1.透平等熵损失", watt_to_mw, "#1燃机透平等熵损失"),
    "SKY.GT2.RXL_100": ("GT2.燃机热效率", ratio_to_percent, "#2燃机热效率"),
    "SKY.GT2.CMP.DSXL_100": ("GT2.压气机等熵效率", ratio_to_percent, "#2燃机压气机等熵效率"),
    "SKY.GT2.CMP.HG_MW": ("GT2.压气机耗功", watt_to_mw, "#2燃机压气机耗功"),
    "SKY.GT2.COMB.T3_K": ("GT2.燃烧室温度", identity, "#2燃机燃烧室燃烧温度"),
    "SKY.GT2.RH_MW": ("GT2.燃料能量", watt_to_mw, "#2燃机燃料热输入/热耗"),
    "SKY.GT2.TURB.DSXL_100": ("GT2.透平等熵效率", ratio_to_percent, "#2燃机透平等熵效率"),
    "SKY.GT2.TURB.YXL_100": ("GT2.透平㶲效率", ratio_to_percent, "#2燃机透平㶲效率"),
    "SKY.GT2.TURB.ZG_MW": ("GT2.透平输出功", watt_to_mw, "#2燃机透平做功"),
    "SKY.GT2.CMP.DSSS_MW": ("GT2.压气机等熵损失", watt_to_mw, "#2燃机压气机等熵损失"),
    "SKY.GT2.TURB.DSSS_MW": ("GT2.透平等熵损失", watt_to_mw, "#2燃机透平等熵损失"),
    "SKY.HRSG1.HRXL_100": ("HRSG1.余热锅炉换热效率", ratio_to_percent, "#1余热锅炉换热效率"),
    "SKY.HRSG1.YXL_100": ("HRSG1.余热锅炉㶲效率", ratio_to_percent, "#1余热锅炉㶲效率"),
    "SKY.HRSG1.HST.XRL_MW": ("HRSG1.主蒸汽吸热功率", watt_to_mw, "#1余热锅炉主蒸汽吸热功率"),
    "SKY.HRSG1.RST.XRL_MW": ("HRSG1.中压+再热蒸汽吸热功率", watt_to_mw, "#1余热锅炉再热蒸汽吸热功率"),
    "SKY.HRSG1.LST.XRL_MW": ("HRSG1.低压主蒸汽功率", watt_to_mw, "#1余热锅炉低压主蒸汽吸热功率"),
    "SKY.HRSG1.YQRL_MW": ("HRSG1.烟气放热功率", watt_to_mw, "#1余热锅炉烟气放热功率"),
    "SKY.HRSG2.HRXL_100": ("HRSG2.余热锅炉换热效率", ratio_to_percent, "#2余热锅炉换热效率"),
    "SKY.HRSG2.YXL_100": ("HRSG2.余热锅炉㶲效率", ratio_to_percent, "#2余热锅炉㶲效率"),
    "SKY.HRSG2.HST.XRL_MW": ("HRSG2.主蒸汽吸热功率", watt_to_mw, "#2余热锅炉主蒸汽吸热功率"),
    "SKY.HRSG2.RST.XRL_MW": ("HRSG2.中压+再热蒸汽吸热功率", watt_to_mw, "#2余热锅炉再热蒸汽吸热功率"),
    "SKY.HRSG2.LST.XRL_MW": ("HRSG2.低压主蒸汽功率", watt_to_mw, "#2余热锅炉低压主蒸汽吸热功率"),
    "SKY.HRSG2.YQRL_MW": ("HRSG2.烟气放热功率", watt_to_mw, "#2余热锅炉烟气放热功率"),
    "SKY.HRSG1.GYZFQ.HRXL_100": (None, identity, "#1余热锅炉高压蒸发器换热效率"),
    "SKY.HRSG1.GYZFQ.HRWC": (None, identity, "#1余热锅炉高压蒸发器换热温差"),
    "SKY.HRSG1.GYZFQ.XRGL": (None, identity, "#1余热锅炉高压蒸发器吸热功率"),
    "SKY.HRSG1.GYZFQ.HRXN": (None, identity, "#1余热锅炉高压蒸发器换热性能系数"),
    "SKY.HRSG1.ZYZFQ.HRXL_100": (None, identity, "#1余热锅炉中压蒸发器换热效率"),
    "SKY.HRSG1.ZYZFQ.HRWC": (None, identity, "#1余热锅炉中压蒸发器换热温差"),
    "SKY.HRSG1.ZYZFQ.XRGL": (None, identity, "#1余热锅炉中压蒸发器吸热功率"),
    "SKY.HRSG1.ZYZFQ.HRXN": (None, identity, "#1余热锅炉中压蒸发器换热性能系数"),
    "SKY.HRSG2.GYZFQ.HRXL_100": (None, identity, "#2余热锅炉高压蒸发器换热效率"),
    "SKY.HRSG2.GYZFQ.HRWC": (None, identity, "#2余热锅炉高压蒸发器换热温差"),
    "SKY.HRSG2.GYZFQ.XRGL": (None, identity, "#2余热锅炉高压蒸发器吸热功率"),
    "SKY.HRSG2.GYZFQ.HRXN": (None, identity, "#2余热锅炉高压蒸发器换热性能系数"),
    "SKY.HRSG2.ZYZFQ.HRXL_100": (None, identity, "#2余热锅炉中压蒸发器换热效率"),
    "SKY.HRSG2.ZYZFQ.HRWC": (None, identity, "#2余热锅炉中压蒸发器换热温差"),
    "SKY.HRSG2.ZYZFQ.XRGL": (None, identity, "#2余热锅炉中压蒸发器吸热功率"),
    "SKY.HRSG2.ZYZFQ.HRXN": (None, identity, "#2余热锅炉中压蒸发器换热性能系数"),
    "SKY.ST.DSXL_100": ("ST.汽机总体等熵效率", ratio_to_percent, "汽轮机整体等熵效率"),
    "SKY.ST.HST.DSXL_100": ("ST.高压缸等熵效率", ratio_to_percent, "汽轮机高压缸等熵效率"),
    "SKY.ST.HST.YXL_100": ("ST.高压缸㶲效率", ratio_to_percent, "汽轮机高压缸㶲效率"),
    "SKY.ST.HST.ZG_MW": ("ST.高压缸出力", watt_to_mw, "汽轮机高压缸出力"),
    "SKY.ST.HST.ZZQLL_kg_s": ("ST.主蒸汽流量", identity, "汽轮机主蒸汽流量"),
    "SKY.ST.IST.DSXL_100": ("ST.中压缸等熵效率", ratio_to_percent, "汽轮机中压缸等熵效率"),
    "SKY.ST.IST.YXL_100": ("ST.中压缸㶲效率", ratio_to_percent, "汽轮机中压缸㶲效率"),
    "SKY.ST.IST.ZG_MW": ("ST.中压缸出力", watt_to_mw, "汽轮机中压缸出力"),
    "SKY.ST.IST.ZRQLL_kg_s": ("ST.再热蒸汽流量", identity, "汽轮机再热蒸汽流量"),
    "SKY.ST.LST.DSXL_100": ("ST.低压缸等熵效率", ratio_to_percent, "汽轮机低压缸等熵效率"),
    "SKY.ST.LST.YXL_100": ("ST.低压缸㶲效率", ratio_to_percent, "汽轮机低压缸㶲效率"),
    "SKY.ST.LST.ZG_MW": (("ST.低压缸出力", "ST.低压缸出力（反算）"), watt_to_mw, "汽轮机低压缸出力"),
}


def is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def safe_div(a, b):
    if is_missing(a) or is_missing(b) or b == 0:
        return None
    try:
        return a / b
    except Exception:
        return None


def safe_sum(values):
    values = [value for value in values if not is_missing(value)]
    return sum(values) if values else None


def safe_sum_required(values):
    if any(is_missing(value) for value in values):
        return None
    return sum(values)


def safe_sub(a, b):
    if is_missing(a) or is_missing(b):
        return None
    try:
        return a - b
    except Exception:
        return None


def safe_one_minus(*values):
    if any(is_missing(value) for value in values):
        return None
    try:
        return 1 - sum(values)
    except Exception:
        return None


def normalize_metric_value(value):
    if is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_metrics(metrics):
    return {key: normalize_metric_value(value) for key, value in metrics.items()}


def output_value(metrics, source, transform):
    if source is None:
        return None
    if isinstance(source, tuple):
        value = first_available(metrics, source)
    else:
        value = metrics.get(source)
    value = normalize_metric_value(value)
    if value is None:
        return None
    return normalize_metric_value(transform(value))


def build_output_json(metrics, timestamp=None, period=1):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_point = []
    for varname, (source, transform, _meaning) in OUTPUT_MAPPING.items():
        result_point.append(
            {
                "varname": varname,
                "timestamp": timestamp,
                "value": output_value(metrics, source, transform),
                "period": period,
            }
        )
    return {
        "result_point": result_point,
        "event_key": PERFORMANCE_EVENT_KEY,
    }


def build_output_mapping_comments():
    comments = []
    for varname, (source, _transform, meaning) in OUTPUT_MAPPING.items():
        if source is None:
            source_metric = None
        elif isinstance(source, tuple):
            source_metric = list(source)
        else:
            source_metric = source
        comments.append(
            {
                "varname": varname,
                "meaning": meaning,
                "source_metric": source_metric,
            }
        )
    return comments


def write_output_json(metrics, output_path=JSON_OUTPUT_PATH, timestamp=None, period=1):
    payload = build_output_json(metrics, timestamp=timestamp, period=period)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def write_output_mapping_comments(output_path=None):
    if output_path is None:
        output_path = JSON_OUTPUT_PATH.with_name("output_mapping_comments.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_output_mapping_comments(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def flue_gas_flow(row, unit):
    return safe_sum_required([
        row[f"燃料质量流量_{unit}"],
        row[f"燃机进口空气流量_{unit}"],
    ])


def gt_fuel_flow(row, unit):
    return row[f"燃料质量流量_{unit}"]




def is_gt_running(row, unit):
    power = row[f"燃机出力_{unit}"]
    fuel_flow = gt_fuel_flow(row, unit)
    gas_flow = flue_gas_flow(row, unit)
    if any(is_missing(value) for value in (power, fuel_flow, gas_flow)):
        return False
    return (
        power > GT_POWER_STOP_THRESHOLD
        and fuel_flow > GT_FUEL_FLOW_STOP_THRESHOLD
        and gas_flow > GT_FLUE_GAS_FLOW_STOP_THRESHOLD_LOWER
        and gas_flow < GT_FLUE_GAS_FLOW_STOP_THRESHOLD_UPPER
    )


def is_lp_cut(row):
    pressure = row["低压缸进汽压力"]
    return False if is_missing(pressure) else pressure < LP_CUT_PRESSURE_THRESHOLD


def heat_absorbed(streams, inlet_key, outlet_key):
    inlet_energy = streams[inlet_key].energy_flow
    outlet_energy = streams[outlet_key].energy_flow
    return safe_sub(outlet_energy, inlet_energy)


def gt_actual_power(row, unit):
    return row[f"燃机出力_{unit}"]


def gt_fuel_energy(gt):
    if gt is None or is_missing(gt.fuel.m_dot):
        return None
    try:
        return gt.chamber.fuel_lhv * gt.fuel.m_dot
    except Exception:
        return None


def add_gt_metrics(metrics, row, gt, unit):
    prefix = f"GT{unit}"
    if gt is None:
        metrics[f"{prefix}.是否运行"] = False
        metrics[f"{prefix}.烟气流量"] = flue_gas_flow(row, unit)
        metrics[f"{prefix}.燃机热效率"] = None
        metrics[f"{prefix}.燃机实际功"] = None
        metrics[f"{prefix}.燃机计算功"] = None
        metrics[f"{prefix}.燃烧室温度"] = None
        metrics[f"{prefix}.压气机等熵效率"] = None
        metrics[f"{prefix}.压气机㶲效率"] = None
        metrics[f"{prefix}.透平等熵效率"] = None
        metrics[f"{prefix}.透平㶲效率"] = None
        metrics[f"{prefix}.压气机等熵损失"] = None
        metrics[f"{prefix}.透平等熵损失"] = None
        metrics[f"{prefix}.压气机耗功"] = None
        metrics[f"{prefix}.透平输出功"] = None
        metrics[f"{prefix}.燃料流量"] = None
        metrics[f"{prefix}.空气流量"] = None
        metrics[f"{prefix}.燃料能量"] = None
        metrics[f"{prefix}.发电功率占比"] = None
        metrics[f"{prefix}.压气机耗功占比"] = None
        metrics[f"{prefix}.排气热量和其他损失占比"] = None
        return

    actual_power = gt_actual_power(row, unit)
    fuel_energy = gt_fuel_energy(gt)

    metrics[f"{prefix}.是否运行"] = True
    metrics[f"{prefix}.烟气流量"] = flue_gas_flow(row, unit)
    metrics[f"{prefix}.燃机热效率"] = safe_div(actual_power, fuel_energy)
    metrics[f"{prefix}.燃机实际功"] = actual_power
    try:
        metrics[f"{prefix}.燃机计算功"] = gt.net_power
    except Exception:
        metrics[f"{prefix}.燃机计算功"] = None
    try:
        metrics[f"{prefix}.燃烧室温度"] = gt.chamber.state_3.T
    except Exception:
        metrics[f"{prefix}.燃烧室温度"] = None
    for metric_name, getter in (
        ("压气机等熵效率", lambda: gt.compressor.isentropic_efficiency),
        ("压气机㶲效率", lambda: gt.compressor.exergy_efficiency),
        ("透平等熵效率", lambda: gt.turbine.isentropic_efficiency),
        ("透平㶲效率", lambda: gt.turbine.exergy_efficiency),
        ("压气机等熵损失", lambda: gt.compressor.isentropic_loss),
        ("透平等熵损失", lambda: gt.turbine.isentropic_loss),
        ("压气机耗功", lambda: gt.compressor.power),
        ("透平输出功", lambda: gt.turbine.power),
    ):
        try:
            metrics[f"{prefix}.{metric_name}"] = getter()
        except Exception:
            metrics[f"{prefix}.{metric_name}"] = None
    metrics[f"{prefix}.燃料流量"] = gt.fuel.m_dot
    metrics[f"{prefix}.空气流量"] = gt.state_1.m_dot
    metrics[f"{prefix}.燃料能量"] = fuel_energy
    metrics[f"{prefix}.发电功率占比"] = safe_div(actual_power, fuel_energy)
    metrics[f"{prefix}.压气机耗功占比"] = safe_div(gt.compressor.power, fuel_energy)
    metrics[f"{prefix}.排气热量和其他损失占比"] = safe_one_minus(
        metrics[f"{prefix}.发电功率占比"],
        metrics[f"{prefix}.压气机耗功占比"],
    )


def st_cylinder_efficiency(turbine):
    try:
        return turbine.isentropic_efficiency[0]
    except Exception:
        return None


def st_cylinder_exergy_efficiency(turbine):
    try:
        return turbine.exergy_efficiency
    except Exception:
        return None


def st_cylinder_mass_flow(turbine):
    return turbine.inlets[0].m_dot


def st_cylinder_ideal_power(turbine):
    inlet = turbine.inlets[0]
    outlet_s = turbine.ideal_outlet_state
    if is_missing(inlet.m_dot) or is_missing(inlet.h) or is_missing(outlet_s.h):
        return None
    return inlet.m_dot * (inlet.h - outlet_s.h)


def st_total_power(plant, lp_cut=False):
    power = safe_sum_required([plant["hp_turbine"].power_output, plant["ip_turbine"].power_output])
    if power is None:
        return None
    if not lp_cut:
        power = safe_sum_required([power, plant["lp_turbine"].power_output])
    return power


def st_total_isentropic_efficiency(plant, actual_power=None, lp_cut=False):
    if actual_power is None:
        actual_power = st_total_power(plant, lp_cut)
    ideal_power = safe_sum_required([
        st_cylinder_ideal_power(plant["hp_turbine"]),
        st_cylinder_ideal_power(plant["ip_turbine"]),
    ])
    if ideal_power is None:
        return None
    if not lp_cut:
        ideal_power = safe_sum_required([ideal_power, st_cylinder_ideal_power(plant["lp_turbine"])])
    return safe_div(actual_power, ideal_power)


def add_st_metrics(metrics, row, plant, lp_cut=False):
    calculated_power = st_total_power(plant, lp_cut)
    actual_power = row["汽机出力"]
    hp_power = plant["hp_turbine"].power_output
    ip_power = plant["ip_turbine"].power_output
    back_calculated_lp_power = safe_sub(safe_sub(actual_power, hp_power), ip_power)

    metrics["ST.汽轮机功率"] = actual_power
    metrics["ST.汽轮机计算功率"] = calculated_power
    metrics["ST.汽机总体等熵效率"] = st_total_isentropic_efficiency(plant, actual_power, lp_cut)
    metrics["ST.高压缸等熵效率"] = st_cylinder_efficiency(plant["hp_turbine"])
    metrics["ST.中压缸等熵效率"] = st_cylinder_efficiency(plant["ip_turbine"])
    metrics["ST.高压缸㶲效率"] = st_cylinder_exergy_efficiency(plant["hp_turbine"])
    metrics["ST.中压缸㶲效率"] = st_cylinder_exergy_efficiency(plant["ip_turbine"])
    metrics["ST.高压缸出力"] = hp_power
    metrics["ST.中压缸出力"] = ip_power
    metrics["ST.主蒸汽流量"] = st_cylinder_mass_flow(plant["hp_turbine"])
    metrics["ST.再热蒸汽流量"] = st_cylinder_mass_flow(plant["ip_turbine"])
    if lp_cut:
        metrics["ST.低压缸等熵效率"] = None
        metrics["ST.低压缸㶲效率"] = None
        metrics["ST.低压缸出力"] = None
        metrics["ST.低压缸出力（正算）"] = None
        metrics["ST.低压缸出力（反算）"] = None
    else:
        metrics["ST.低压缸等熵效率"] = st_cylinder_efficiency(plant["lp_turbine"])
        metrics["ST.低压缸㶲效率"] = st_cylinder_exergy_efficiency(plant["lp_turbine"])
        metrics["ST.低压缸出力"] = plant["lp_turbine"].power_output
        metrics["ST.低压缸出力（正算）"] = plant["lp_turbine"].power_output
        metrics["ST.低压缸出力（反算）"] = back_calculated_lp_power


def add_st_none_metrics(metrics):
    metrics["ST.汽轮机功率"] = None
    metrics["ST.汽轮机计算功率"] = None
    metrics["ST.汽机总体等熵效率"] = None
    metrics["ST.高压缸等熵效率"] = None
    metrics["ST.中压缸等熵效率"] = None
    metrics["ST.低压缸等熵效率"] = None
    metrics["ST.高压缸㶲效率"] = None
    metrics["ST.中压缸㶲效率"] = None
    metrics["ST.低压缸㶲效率"] = None
    metrics["ST.高压缸出力"] = None
    metrics["ST.中压缸出力"] = None
    metrics["ST.低压缸出力"] = None
    metrics["ST.低压缸计算出力"] = None
    metrics["ST.主蒸汽流量"] = None
    metrics["ST.再热蒸汽流量"] = None


def main_steam_heat_power(streams, unit):
    return heat_absorbed(streams, f"{unit}号炉低压汽包高压给水泵后", f"{unit}号炉高压主蒸汽")


def ip_reheat_steam_heat_power(streams, unit):
    feedwater_heat = streams[f"{unit}号炉低压汽包中压给水泵后"].energy_flow
    cold_reheat_heat = streams[f"{unit}号炉高压缸排汽"].energy_flow
    hot_reheat_heat = streams[f"{unit}号炉热再热出口"].energy_flow
    return safe_sub(safe_sub(hot_reheat_heat, feedwater_heat), cold_reheat_heat)


def lp_main_steam_heat_power(streams, unit):
    return (
        safe_sum_required([
            heat_absorbed(streams, f"{unit}号炉低压省煤器入口", f"{unit}号炉低压省煤器出口"),
            heat_absorbed(streams, f"{unit}号炉低压汽包给水调节阀出口", f"{unit}号炉低压汽包"),
            heat_absorbed(streams, f"{unit}号炉低压汽包至低压主蒸汽", f"{unit}号炉低压主蒸汽"),
        ])
    )


def add_hrsg_metrics(metrics, plant, streams, unit, is_running):
    prefix = f"HRSG{unit}"
    if not is_running:
        metrics[f"{prefix}.是否运行"] = False
        metrics[f"{prefix}.余热锅炉换热效率"] = None
        metrics[f"{prefix}.余热锅炉换热器有效性"] = None
        metrics[f"{prefix}.余热锅炉㶲效率"] = None
        metrics[f"{prefix}.主蒸汽吸热功率"] = None
        metrics[f"{prefix}.中压+再热蒸汽吸热功率"] = None
        metrics[f"{prefix}.低压主蒸汽功率"] = None
        metrics[f"{prefix}.烟气放热功率"] = None
        return

    metrics[f"{prefix}.是否运行"] = True
    hrsg = plant[f"hrsg{unit}"]
    for metric_name, getter in (
        ("余热锅炉换热效率", lambda: hrsg.energy_balance_ratio),
        ("余热锅炉换热器有效性", lambda: hrsg.energy_effectiveness),
        ("余热锅炉㶲效率", lambda: hrsg.exergy_balance_ratio),
        ("主蒸汽吸热功率", lambda: main_steam_heat_power(streams, unit)),
        ("中压+再热蒸汽吸热功率", lambda: ip_reheat_steam_heat_power(streams, unit)),
        ("低压主蒸汽功率", lambda: lp_main_steam_heat_power(streams, unit)),
        ("烟气放热功率", lambda: hrsg.gas_energy_release),
    ):
        try:
            metrics[f"{prefix}.{metric_name}"] = getter()
        except Exception:
            metrics[f"{prefix}.{metric_name}"] = None


def heating_power(streams):
    return streams["热网抽汽"].energy_flow


def heating_exergy(streams):
    return streams["热网抽汽"].exergy_flow


def cal_property(df, idx):
    row = df.iloc[idx]
    gt1_running = is_gt_running(row, 1)
    gt2_running = is_gt_running(row, 2)
    lp_cut = is_lp_cut(row)
    metrics = {"idx": idx}

    if not gt1_running and not gt2_running:
        add_gt_metrics(metrics, row, None, 1)
        add_gt_metrics(metrics, row, None, 2)
        add_st_none_metrics(metrics)
        add_hrsg_metrics(metrics, None, None, 1, False)
        add_hrsg_metrics(metrics, None, None, 2, False)
        metrics["PLANT.联合循环发电功率"] = None
        metrics["PLANT.联合循环发电效率"] = None
        metrics["PLANT.联合循环能量效率"] = None
        metrics["PLANT.联合循环热电效率"] = None
        metrics["PLANT.联合循环㶲效率"] = None
        metrics["PLANT.热耗"] = None
        metrics["PLANT.燃料能量"] = None
        metrics["PLANT.供热㶲"] = None
        metrics["PLANT.燃机功率"] = None
        metrics["PLANT.汽轮机实际功率"] = None
        metrics["PLANT.汽轮机计算功率"] = None
        metrics["PLANT.供热"] = None
        return normalize_metrics(metrics)

    streams = build_streams_from_row(df, idx)
    plant = build_plant(df, idx)
    gt1 = build_gt(df, idx, 1) if gt1_running else None
    gt2 = build_gt(df, idx, 2) if gt2_running else None

    add_gt_metrics(metrics, row, gt1, 1)
    add_gt_metrics(metrics, row, gt2, 2)
    add_st_metrics(metrics, row, plant, lp_cut)
    add_hrsg_metrics(metrics, plant, streams, 1, gt1 is not None)
    add_hrsg_metrics(metrics, plant, streams, 2, gt2 is not None)

    running_gts = [(unit, gt) for unit, gt in ((1, gt1), (2, gt2)) if gt is not None]
    gt_power = safe_sum([gt_actual_power(row, unit) for unit, gt in running_gts])
    st_power = row["汽机出力"]
    fuel_energy = safe_sum([gt_fuel_energy(gt) for unit, gt in running_gts])
    generation_power = safe_sum([gt_power, st_power])
    heat_power = df.iloc[idx]["热网换热量"]
    heat_exergy = heating_exergy(streams)

    metrics["PLANT.联合循环发电功率"] = generation_power
    metrics["PLANT.联合循环发电效率"] = safe_div(generation_power, fuel_energy)
    metrics["PLANT.联合循环热电效率"] = safe_div(safe_sum_required([generation_power, heat_power]), fuel_energy)
    metrics["PLANT.联合循环能量效率"] = metrics["PLANT.联合循环热电效率"]
    metrics["PLANT.联合循环㶲效率"] = safe_div(safe_sum_required([generation_power, heat_exergy]), fuel_energy)
    metrics["PLANT.热耗"] = fuel_energy
    metrics["PLANT.燃料能量"] = fuel_energy
    metrics["PLANT.供热㶲"] = heat_exergy
    metrics["PLANT.燃机功率"] = gt_power
    metrics["PLANT.汽轮机实际功率"] = row["汽机出力"]
    metrics["PLANT.汽轮机计算功率"] = metrics["ST.汽轮机计算功率"]
    metrics["PLANT.供热"] = heat_power

    return normalize_metrics(metrics)


# if __name__ == "__main__":
#     df = pd.read_csv(DATA_PATH)
#     sample_count = min(1000, len(df))
#     indexes = np.linspace(0, len(df) - 1, sample_count, dtype=int)
#     results = []
#     total_start = time.perf_counter() 

#     for i, idx in enumerate(indexes, start=1):
#         point_start = time.perf_counter()
#         print(f"计算第 {i}/{sample_count} 个点: idx={idx}")
#         try:
#             results.append(cal_property(df, idx))
#         except Exception as exc:
#             results.append({"idx": idx, "error": str(exc)})
#         point_time = time.perf_counter() - point_start
#         elapsed = time.perf_counter() - total_start
#         avg_time = elapsed / i
#         remaining_time = avg_time * (sample_count - i)
#         print(
#             f"idx={idx} 耗时 {point_time:.2f}s, "
#             f"已用 {elapsed:.2f}s, "
#             f"平均 {avg_time:.2f}s/点, "
#             f"预计剩余 {remaining_time:.2f}s"
#         )

#         if i % 500 == 0:
#             OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#             pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
#     total_time = time.perf_counter() - total_start
#     print(f"总耗时: {total_time:.2f}s, 平均耗时: {total_time / sample_count:.2f}s/点")
#     print(f"结果已保存: {OUTPUT_PATH}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GTCC performance calculation from an input JSON file.",
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help="Path to the input JSON file. Defaults to input/example/input.json.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to calculate when the JSON contains multiple frames.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the computed output JSON. Defaults to output/<input_json_name>_outcome.json.",
    )
    parser.add_argument(
        "--comments-output",
        default=None,
        help="Optional path for output mapping comments JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output) if args.output else default_output_path_for_input(input_path)

    df, timestamp, diagnostics = build_online_dataframe_from_file(
        input_path,
        frame_index=args.frame_index,
    )
    results = cal_property(df, 0)
    write_output_json(results, output_path=output_path, timestamp=timestamp)
    print(f"input_json={input_path}")
    print(f"frame_index={args.frame_index}")
    print(f"timestamp={timestamp}")
    print(f"missing_internal_fields={len(diagnostics['missing_internal_fields'])}")
    print(f"JSON result saved: {output_path}")
    if args.comments_output:
        comments_path = write_output_mapping_comments(args.comments_output)
        print(f"Output mapping comments saved: {comments_path}")


if __name__ == "__main__":
    main()
