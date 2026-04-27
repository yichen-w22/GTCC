import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Energy_Utilization_Analysis.energy_analysis.GT_model.compressor import build_isentropic_process, sample_isentropic_process
from Energy_Utilization_Analysis.energy_analysis.working_fluid.streams import build_gases_from_row


DATA_PATH = Path(
    r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\continuous_data_10min.csv"
)

rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


def get_compressor_states(df, idx, unit):
    gases = list(build_gases_from_row(df, idx).values())

    if unit == 1:
        inlet = gases[2]
        outlet = gases[4]
    elif unit == 2:
        inlet = gases[3]
        outlet = gases[5]
    else:
        raise ValueError("unit must be 1 or 2")

    return inlet, outlet


def check_process(inlet, outlet, label):
    result_0 = build_isentropic_process(inlet, outlet, 0.0)
    result_05 = build_isentropic_process(inlet, outlet, 0.5)
    result_1 = build_isentropic_process(inlet, outlet, 1.0)

    print(label)
    print("-" * 60)
    print(f"process_type = {result_05.process_type}")
    print(f"efficiency = {result_05.efficiency:.6f}")
    print(f"delta_h = {result_05.delta_h:.6f}")
    print(f"energy_change = {result_05.energy_change:.6f}")
    print(f"state@0.0: T={result_0.state_at_progress.T:.6f}, P={result_0.state_at_progress.P:.6f}, h={result_0.state_at_progress.h:.6f}")
    print(f"state@0.5: T={result_05.state_at_progress.T:.6f}, P={result_05.state_at_progress.P:.6f}, h={result_05.state_at_progress.h:.6f}")
    print(f"state@1.0: T={result_1.state_at_progress.T:.6f}, P={result_1.state_at_progress.P:.6f}, h={result_1.state_at_progress.h:.6f}")

    assert result_05.process_type == "compression"
    assert result_05.efficiency > 0.0
    assert result_05.delta_h > 0.0
    assert result_05.energy_change is not None and result_05.energy_change > 0.0

    assert abs(result_0.state_at_progress.h - inlet.h) / max(abs(inlet.h), 1.0) < 1e-5
    assert abs(result_1.state_at_progress.h - outlet.h) / max(abs(outlet.h), 1.0) < 1e-5
    assert inlet.h < result_05.state_at_progress.h < outlet.h
    assert inlet.P < result_05.state_at_progress.P < outlet.P
    assert result_05.outlet_isentropic.P == outlet.P


def plot_ts_process(inlet, outlet, label, output_path):
    samples = sample_isentropic_process(inlet, outlet, n_points=21)
    s_curve = [item.state_at_progress.s for item in samples]
    t_curve = [item.state_at_progress.T for item in samples]
    progress_points = [0.25, 0.50, 0.75]
    progress_results = [build_isentropic_process(inlet, outlet, progress) for progress in progress_points]
    outlet_isentropic = samples[-1].outlet_isentropic

    plt.figure(figsize=(9, 6))
    plt.plot(s_curve, t_curve, color="#c0392b", linewidth=2.0, label="实际过程")
    plt.plot(
        [inlet.s, outlet_isentropic.s],
        [inlet.T, outlet_isentropic.T],
        color="#2980b9",
        linestyle="--",
        linewidth=1.8,
        label="等熵参考线",
    )
    plt.scatter([inlet.s], [inlet.T], color="black", label="入口", zorder=3)
    plt.scatter([outlet.s], [outlet.T], color="#c0392b", label="出口", zorder=3)
    plt.scatter([outlet_isentropic.s], [outlet_isentropic.T], color="#2980b9", label="等熵出口", zorder=3)

    for progress, result in zip(progress_points, progress_results):
        plt.scatter([result.state_at_progress.s], [result.state_at_progress.T], zorder=4)
        plt.text(
            result.state_at_progress.s,
            result.state_at_progress.T,
            f"  x={progress:.2f}",
            fontsize=9,
        )

    plt.xlabel("s / J/(kg.K)")
    plt.ylabel("T / K")
    plt.title(label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)
    idx = 1000

    inlet_1, outlet_1 = get_compressor_states(df, idx, 1)
    inlet_2, outlet_2 = get_compressor_states(df, idx, 2)

    check_process(inlet_1, outlet_1, "GT1 compressor isentropic-process test")
    plot_ts_process(
        inlet_1,
        outlet_1,
        "GT1 compressor T-s process",
        CURRENT_DIR / "gt1_compressor_ts.png",
    )
    print()
    check_process(inlet_2, outlet_2, "GT2 compressor isentropic-process test")
    plot_ts_process(
        inlet_2,
        outlet_2,
        "GT2 compressor T-s process",
        CURRENT_DIR / "gt2_compressor_ts.png",
    )
    print()
    print("PASS")


if __name__ == "__main__":
    main()
