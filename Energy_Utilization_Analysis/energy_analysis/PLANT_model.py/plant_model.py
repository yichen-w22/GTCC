import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.GT_model.GT_model import build_gt
from energy_analysis.ST_model.plant import build_plant
from energy_analysis.working_fluid.streams import build_streams_from_row

DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
OUTPUT_DIR = PROJECT_ROOT / "temp"
OUTPUT_PATH = OUTPUT_DIR / "plant_metrics.csv"
GT_POWER_STOP_THRESHOLD = 1.0e6
GT_FUEL_FLOW_STOP_THRESHOLD = 5.0
GT_FLUE_GAS_FLOW_STOP_THRESHOLD_LOWER = 100.0
GT_FLUE_GAS_FLOW_STOP_THRESHOLD_UPPER = 1000.0
LP_CUT_PRESSURE_THRESHOLD = 0.5 * 101325.0


def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def safe_sum(values):
    values = [value for value in values if value is not None and not pd.isna(value)]
    return sum(values) if values else None


def flue_gas_flow(row, unit):
    return row[f"燃料质量流量_{unit}"] + row[f"燃机进口空气流量_{unit}"]


def gt_fuel_flow(row, unit):
    return row[f"燃料质量流量_{unit}"]




def is_gt_running(row, unit):
    return (
        row[f"燃机出力_{unit}"] > GT_POWER_STOP_THRESHOLD
        and gt_fuel_flow(row, unit) > GT_FUEL_FLOW_STOP_THRESHOLD
        and flue_gas_flow(row, unit) > GT_FLUE_GAS_FLOW_STOP_THRESHOLD_LOWER
        and flue_gas_flow(row, unit) < GT_FLUE_GAS_FLOW_STOP_THRESHOLD_UPPER
    )


def is_lp_cut(row):
    return row["低压缸进汽压力"] < LP_CUT_PRESSURE_THRESHOLD


def heat_absorbed(streams, inlet_key, outlet_key):
    return streams[outlet_key].energy_flow - streams[inlet_key].energy_flow


def gt_actual_power(row, unit):
    return row[f"燃机出力_{unit}"]


def gt_fuel_energy(gt):
    return gt.chamber.fuel_lhv * gt.fuel.m_dot


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
    metrics[f"{prefix}.燃机计算功"] = gt.net_power
    metrics[f"{prefix}.燃烧室温度"] = gt.chamber.state_3.T
    metrics[f"{prefix}.压气机等熵效率"] = gt.compressor.isentropic_efficiency
    metrics[f"{prefix}.压气机㶲效率"] = gt.compressor.exergy_efficiency
    metrics[f"{prefix}.透平等熵效率"] = gt.turbine.isentropic_efficiency
    metrics[f"{prefix}.透平㶲效率"] = gt.turbine.exergy_efficiency
    metrics[f"{prefix}.压气机耗功"] = gt.compressor.power
    metrics[f"{prefix}.透平输出功"] = gt.turbine.power
    metrics[f"{prefix}.燃料流量"] = gt.fuel.m_dot
    metrics[f"{prefix}.空气流量"] = gt.state_1.m_dot
    metrics[f"{prefix}.燃料能量"] = fuel_energy
    metrics[f"{prefix}.发电功率占比"] = safe_div(actual_power, fuel_energy)
    metrics[f"{prefix}.压气机耗功占比"] = safe_div(gt.compressor.power, fuel_energy)
    metrics[f"{prefix}.排气热量和其他损失占比"] = 1 - safe_div(actual_power, fuel_energy) - safe_div(gt.compressor.power, fuel_energy)


def st_cylinder_efficiency(turbine):
    return turbine.isentropic_efficiency[0]


def st_cylinder_ideal_power(turbine):
    inlet = turbine.inlets[0]
    outlet_s = turbine.ideal_outlet_state
    return inlet.m_dot * (inlet.h - outlet_s.h)


def st_total_power(plant, lp_cut=False):
    power = plant["hp_turbine"].power_output + plant["ip_turbine"].power_output
    if not lp_cut:
        power += plant["lp_turbine"].power_output
    return power


def st_total_isentropic_efficiency(plant, actual_power=None, lp_cut=False):
    if actual_power is None:
        actual_power = st_total_power(plant, lp_cut)
    ideal_power = (
        st_cylinder_ideal_power(plant["hp_turbine"])
        + st_cylinder_ideal_power(plant["ip_turbine"])
    )
    if not lp_cut:
        ideal_power += st_cylinder_ideal_power(plant["lp_turbine"])
    return safe_div(actual_power, ideal_power)


def add_st_metrics(metrics, row, plant, lp_cut=False):
    calculated_power = st_total_power(plant, lp_cut)
    actual_power = row["汽机出力"]
    hp_power = plant["hp_turbine"].power_output
    ip_power = plant["ip_turbine"].power_output
    back_calculated_lp_power = actual_power - hp_power - ip_power

    metrics["ST.汽轮机功率"] = actual_power
    metrics["ST.汽轮机计算功率"] = calculated_power
    metrics["ST.汽机总体等熵效率"] = st_total_isentropic_efficiency(plant, actual_power, lp_cut)
    metrics["ST.高压缸等熵效率"] = st_cylinder_efficiency(plant["hp_turbine"])
    metrics["ST.中压缸等熵效率"] = st_cylinder_efficiency(plant["ip_turbine"])
    metrics["ST.高压缸出力"] = hp_power
    metrics["ST.中压缸出力"] = ip_power
    if lp_cut:
        metrics["ST.低压缸等熵效率"] = None
        metrics["ST.低压缸出力（正算）"] = None
        metrics["ST.低压缸出力（反算）"] = None
    else:
        metrics["ST.低压缸等熵效率"] = st_cylinder_efficiency(plant["lp_turbine"])
        metrics["ST.低压缸出力（正算）"] = plant["lp_turbine"].power_output
        metrics["ST.低压缸出力（反算）"] = back_calculated_lp_power


def add_st_none_metrics(metrics):
    metrics["ST.汽轮机功率"] = None
    metrics["ST.汽轮机计算功率"] = None
    metrics["ST.汽机总体等熵效率"] = None
    metrics["ST.高压缸等熵效率"] = None
    metrics["ST.中压缸等熵效率"] = None
    metrics["ST.低压缸等熵效率"] = None
    metrics["ST.高压缸出力"] = None
    metrics["ST.中压缸出力"] = None
    metrics["ST.低压缸出力"] = None
    metrics["ST.低压缸计算出力"] = None


def main_steam_heat_power(streams, unit):
    return heat_absorbed(streams, f"{unit}号炉低压汽包高压给水泵后", f"{unit}号炉高压主蒸汽")


def ip_reheat_steam_heat_power(streams, unit):
    feedwater_heat = streams[f"{unit}号炉低压汽包中压给水泵后"].energy_flow
    cold_reheat_heat = streams[f"{unit}号炉高压缸排汽"].energy_flow
    hot_reheat_heat = streams[f"{unit}号炉热再热出口"].energy_flow
    return hot_reheat_heat - feedwater_heat - cold_reheat_heat


def lp_main_steam_heat_power(streams, unit):
    return (
        heat_absorbed(streams, f"{unit}号炉低压省煤器入口", f"{unit}号炉低压省煤器出口")
        + heat_absorbed(streams, f"{unit}号炉低压汽包给水调节阀出口", f"{unit}号炉低压汽包")
        + heat_absorbed(streams, f"{unit}号炉低压汽包至低压主蒸汽", f"{unit}号炉低压主蒸汽")
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

    hrsg = plant[f"hrsg{unit}"]
    metrics[f"{prefix}.是否运行"] = True
    metrics[f"{prefix}.余热锅炉换热效率"] = hrsg.energy_balance_ratio
    metrics[f"{prefix}.余热锅炉换热器有效性"] = hrsg.energy_effectiveness
    metrics[f"{prefix}.余热锅炉㶲效率"] = hrsg.exergy_balance_ratio
    metrics[f"{prefix}.主蒸汽吸热功率"] = main_steam_heat_power(streams, unit)
    metrics[f"{prefix}.中压+再热蒸汽吸热功率"] = ip_reheat_steam_heat_power(streams, unit)
    metrics[f"{prefix}.低压主蒸汽功率"] = lp_main_steam_heat_power(streams, unit)
    metrics[f"{prefix}.烟气放热功率"] = hrsg.gas_energy_release


def heating_power(streams):
    return streams["热网抽汽"].energy_flow


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
        metrics["PLANT.联合循环热电效率"] = None
        metrics["PLANT.燃机功率"] = None
        metrics["PLANT.汽轮机实际功率"] = None
        metrics["PLANT.汽轮机计算功率"] = None
        metrics["PLANT.供热"] = None
        return metrics

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

    metrics["PLANT.联合循环发电功率"] = generation_power
    metrics["PLANT.联合循环发电效率"] = safe_div(generation_power, fuel_energy)
    metrics["PLANT.联合循环热电效率"] = safe_div(safe_sum([generation_power, heat_power]), fuel_energy)
    metrics["PLANT.燃机功率"] = gt_power
    metrics["PLANT.汽轮机实际功率"] = row["汽机出力"]
    metrics["PLANT.汽轮机计算功率"] = metrics["ST.汽轮机计算功率"]
    metrics["PLANT.供热"] = heat_power

    return metrics


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    sample_count = min(10000, len(df))
    indexes = np.linspace(0, len(df) - 1, sample_count, dtype=int)
    results = []
    total_start = time.perf_counter() 

    for i, idx in enumerate(indexes, start=1):
        point_start = time.perf_counter()
        print(f"计算第 {i}/{sample_count} 个点: idx={idx}")
        try:
            results.append(cal_property(df, idx))
        except Exception as exc:
            results.append({"idx": idx, "error": str(exc)})
        point_time = time.perf_counter() - point_start
        elapsed = time.perf_counter() - total_start
        avg_time = elapsed / i
        remaining_time = avg_time * (sample_count - i)
        print(
            f"idx={idx} 耗时 {point_time:.2f}s, "
            f"已用 {elapsed:.2f}s, "
            f"平均 {avg_time:.2f}s/点, "
            f"预计剩余 {remaining_time:.2f}s"
        )

        if i % 500 == 0:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    total_time = time.perf_counter() - total_start
    print(f"总耗时: {total_time:.2f}s, 平均耗时: {total_time / sample_count:.2f}s/点")
    print(f"结果已保存: {OUTPUT_PATH}")

# if __name__ == "__main__":
#     idx = 6000
#     df = pd.read_csv(DATA_PATH)
#     results = cal_property(df, idx)

#     for key, value in results.items():
#         print(f"{key}: {value}")
