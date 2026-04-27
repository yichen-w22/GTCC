import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.GT_model.compressor import Compressor
from Energy_Utilization_Analysis.energy_analysis.working_fluid.streams import build_gases_from_row
from energy_analysis.working_fluid import GasState


DATA_PATH = Path(
    r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\continuous_data_10min.csv"
)


def get_compressor_states(df, idx, unit):
    gases = list(build_gases_from_row(df, idx).values())
    if unit == 1:
        return gases[2], gases[4]
    if unit == 2:
        return gases[3], gases[5]
    raise ValueError("unit must be 1 or 2")


def compressor_diagnosis(inlet_gas, outlet_gas, result):
    k_in = inlet_gas.cp / (inlet_gas.cp - inlet_gas.R)
    pressure_ratio = outlet_gas.P / inlet_gas.P
    temperature_ratio = outlet_gas.T / inlet_gas.T
    t2s_ideal = inlet_gas.T * pressure_ratio ** ((k_in - 1.0) / k_in)
    t2_eta_085 = inlet_gas.T + (t2s_ideal - inlet_gas.T) / 0.85
    t2_eta_090 = inlet_gas.T + (t2s_ideal - inlet_gas.T) / 0.90

    messages = []
    if inlet_gas.P > 3.0e5:
        messages.append(
            "压气机入口压力已经高到 {:.3f} MPa，明显不像燃机进气口，优先怀疑入口压力列取错或单位不对。".format(
                inlet_gas.P / 1e6
            )
        )
    if pressure_ratio < 2.0 and temperature_ratio > 1.5:
        messages.append(
            "压比只有 {:.3f}，但温度比达到 {:.3f}，压力提升很小而温升极大，这和正常压气机过程不一致。".format(
                pressure_ratio, temperature_ratio
            )
        )
    if result.efficiency < 0.3:
        messages.append(
            "等熵效率只有 {:.4f}，主要是因为等熵焓升 {:.1f} J/kg 远小于实际焓升 {:.1f} J/kg。".format(
                result.efficiency,
                result.state_2s.h - inlet_gas.h,
                outlet_gas.h - inlet_gas.h,
            )
        )
    if abs(outlet_gas.T - t2_eta_090) > 120.0:
        messages.append(
            "按入口 cp/R 估算，当前压比下 90% 效率时出口温度大约应在 {:.2f} K；"
            "实测却是 {:.2f} K，偏高 {:.2f} K。".format(
                t2_eta_090,
                outlet_gas.T,
                outlet_gas.T - t2_eta_090,
            )
        )

    return {
        "k_in": k_in,
        "t2s_ideal": t2s_ideal,
        "t2_eta_085": t2_eta_085,
        "t2_eta_090": t2_eta_090,
        "messages": messages,
    }


def check_compressor(inlet_gas, outlet_gas, label):
    compressor = Compressor(name=label)

    result = compressor.solve(
        inlet_gas=inlet_gas,
        outlet_gas=outlet_gas,
        bleeding_mass_fraction=0.05,
        bleeding_pressure_fraction=0.5,
        bleeding_energy_fraction=0.5,
    )

    outlet_same_comp = GasState.from_TP(
        T=outlet_gas.T,
        P=outlet_gas.P,
        m_dot=outlet_gas.m_dot,
        composition=inlet_gas.composition,
        name=f"{label}_outlet_same_comp",
        ref=inlet_gas.ref,
    )

    eta_num = result.state_2s.h - inlet_gas.h
    eta_den = outlet_gas.h - inlet_gas.h
    pressure_ratio = outlet_gas.P / inlet_gas.P
    temperature_ratio = outlet_gas.T / inlet_gas.T
    inlet_comp = inlet_gas.composition.normalized().as_dict()
    outlet_comp = outlet_gas.composition.normalized().as_dict()
    comp_delta = {
        key: outlet_comp[key] - inlet_comp[key]
        for key in inlet_comp
        if abs(outlet_comp[key] - inlet_comp[key]) > 1e-10
    }
    diagnosis = compressor_diagnosis(inlet_gas, outlet_gas, result)

    print(label)
    print("-" * 60)
    print(f"efficiency = {result.efficiency:.6f}")
    print(f"eta numerator   (h2s - h1) = {eta_num:.6f}")
    print(f"eta denominator (h2  - h1) = {eta_den:.6f}")
    print(f"delta_h = {result.delta_h:.6f}")
    print(f"power = {result.power:.6f}")
    print(f"pressure ratio = {pressure_ratio:.6f}")
    print(f"temperature ratio = {temperature_ratio:.6f}")
    print(
        f"inlet : T={inlet_gas.T:.6f}, P={inlet_gas.P:.6f}, "
        f"h={inlet_gas.h:.6f}, s={inlet_gas.s:.6f}, cp={inlet_gas.cp:.6f}, R={inlet_gas.R:.6f}"
    )
    print(
        f"outlet: T={outlet_gas.T:.6f}, P={outlet_gas.P:.6f}, "
        f"h={outlet_gas.h:.6f}, s={outlet_gas.s:.6f}, cp={outlet_gas.cp:.6f}, R={outlet_gas.R:.6f}"
    )
    print(
        f"outlet_same_comp: h={outlet_same_comp.h:.6f}, s={outlet_same_comp.s:.6f}, "
        f"cp={outlet_same_comp.cp:.6f}, R={outlet_same_comp.R:.6f}"
    )
    print(f"inlet composition = {inlet_comp}")
    print(f"outlet composition = {outlet_comp}")
    print(f"composition delta = {comp_delta}")
    if result.bleeding is not None:
        print(
            f"bleeding: m_dot={result.bleeding.m_dot:.6f}, "
            f"P={result.bleeding.P:.6f}, h={result.bleeding.h:.6f}"
        )
    print(
        f"state_2: m_dot={result.state_2.m_dot:.6f}, "
        f"P={result.state_2.P:.6f}, h={result.state_2.h:.6f}"
    )
    print(
        f"state_2s: m_dot={result.state_2s.m_dot:.6f}, "
        f"P={result.state_2s.P:.6f}, h={result.state_2s.h:.6f}"
    )
    print("diagnosis")
    print(
        f"k_in={diagnosis['k_in']:.6f}, "
        f"T2s(ideal-gas estimate)={diagnosis['t2s_ideal']:.6f}, "
        f"T2(eta=0.85 estimate)={diagnosis['t2_eta_085']:.6f}, "
        f"T2(eta=0.90 estimate)={diagnosis['t2_eta_090']:.6f}"
    )
    if diagnosis["messages"]:
        for message in diagnosis["messages"]:
            print(f"- {message}")
    else:
        print("- 未发现明显异常。")

    assert result.efficiency > 0.0
    assert result.delta_h > 0.0
    assert result.power is not None and result.power > 0.0
    assert result.state_2.P == outlet_gas.P
    assert result.state_2.h == outlet_gas.h
    assert result.state_2.m_dot < inlet_gas.m_dot
    assert result.bleeding is not None
    assert result.bleeding.m_dot > 0.0
    assert inlet_gas.P < result.bleeding.P < outlet_gas.P
    assert inlet_gas.h < result.bleeding.h < outlet_gas.h


def main():
    df = pd.read_csv(DATA_PATH)
    idx = 1000

    inlet_1, outlet_1 = get_compressor_states(df, idx, 1)
    check_compressor(inlet_1, outlet_1, "GT1 compressor test")
    print()

    inlet_2, outlet_2 = get_compressor_states(df, idx, 2)
    check_compressor(inlet_2, outlet_2, "GT2 compressor test")
    print()
    print("PASS")


if __name__ == "__main__":
    main()
