import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Energy_Utilization_Analysis.energy_analysis.working_fluid.streams import build_plant_from_streams
from Energy_Utilization_Analysis.energy_analysis.working_fluid.streams import build_gases_from_row, build_streams_from_row


DATA_PATH = Path(
    r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\continuous_data_10min.csv"
)
TARGET_RATIO = 0.99
SAMPLE_SIZE = 100

rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


def build_sample_indices(data_length: int, sample_size: int) -> np.ndarray:
    if data_length <= 0:
        return np.array([], dtype=int)
    if data_length <= sample_size:
        return np.arange(data_length, dtype=int)
    return np.unique(np.linspace(0, data_length - 1, sample_size, dtype=int))


def collect_hrsg_table(data_path: Path, sample_size: int) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    sample_indices = build_sample_indices(len(df), sample_size)
    rows = []
    start = time.time()

    for sample_no, idx in enumerate(sample_indices, start=1):
        try:
            streams = build_streams_from_row(df, idx)
            gases = build_gases_from_row(df, idx)
            plant = build_plant_from_streams(streams, gases)
        except Exception as e:
            print(f"跳过 idx={idx}, error={e}")
            continue

        row = {"time_step": idx}
        valid = True

        for name in ["hrsg1", "hrsg2"]:
            component = plant[name]
            gas_release = component.gas_heat_release()
            water_gain = component.water_heat_absorption()
            ratio = component.heat_balance_ratio()

            if pd.isna(ratio) or pd.isna(gas_release) or pd.isna(water_gain):
                valid = False
                break

            row[f"{name}_gas_heat_release"] = gas_release
            row[f"{name}_water_heat_absorption"] = water_gain
            row[f"{name}_ratio"] = ratio

        if not valid:
            print(f"跳过 idx={idx}, ratio 或换热量为 NaN")
            continue

        rows.append(row)
        elapsed = time.time() - start
        print(f"已计算样本 {sample_no}/{len(sample_indices)}, idx={idx}, 已用时 {elapsed:.2f} s")

    return pd.DataFrame(rows)


def calc_flow_coefficient(table: pd.DataFrame, ratio_column: str, target_ratio: float) -> float:
    avg_ratio = table[ratio_column].mean()
    return avg_ratio / target_ratio


def add_corrected_ratio(table: pd.DataFrame, component_name: str, coeff: float) -> None:
    ratio_col = f"{component_name}_ratio"
    corrected_col = f"{component_name}_ratio_corrected"
    table[corrected_col] = table[ratio_col] / coeff


def plot_ratios(table: pd.DataFrame, coeff_hrsg1: float, coeff_hrsg2: float) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    plot_items = [
        ("hrsg1", "1号余热锅炉", coeff_hrsg1),
        ("hrsg2", "2号余热锅炉", coeff_hrsg2),
    ]

    for ax, (name, title, coeff) in zip(axes, plot_items):
        ax.plot(table["time_step"], table[f"{name}_ratio"], marker="o", label="原始 ratio", linewidth=1.5)
        ax.plot(
            table["time_step"],
            table[f"{name}_ratio_corrected"],
            marker="o",
            label=f"修正后 ratio (系数={coeff:.5f})",
            linewidth=1.5,
        )
        ax.axhline(TARGET_RATIO, color="red", linestyle="--", linewidth=1.0, label="目标 0.99")
        ax.set_ylabel("Heat Balance Ratio")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.show()


def main():
    table = collect_hrsg_table(DATA_PATH, SAMPLE_SIZE)
    if table.empty:
        print("没有得到可用数据点。")
        return

    coeff_hrsg1 = calc_flow_coefficient(table, "hrsg1_ratio", TARGET_RATIO)
    coeff_hrsg2 = calc_flow_coefficient(table, "hrsg2_ratio", TARGET_RATIO)

    add_corrected_ratio(table, "hrsg1", coeff_hrsg1)
    add_corrected_ratio(table, "hrsg2", coeff_hrsg2)

    avg_raw_hrsg1 = table["hrsg1_ratio"].mean()
    avg_raw_hrsg2 = table["hrsg2_ratio"].mean()
    avg_corrected_hrsg1 = avg_raw_hrsg1 / coeff_hrsg1
    avg_corrected_hrsg2 = avg_raw_hrsg2 / coeff_hrsg2

    print("烟气流量修正结果")
    print("-" * 60)
    print(f"1号余热锅炉: 样本点数 = {table['hrsg1_ratio'].notna().sum()}")
    print(f"1号余热锅炉: 原始平均 ratio = {avg_raw_hrsg1:.6f}")
    print(f"1号余热锅炉: 烟气流量修正系数 = {coeff_hrsg1:.6f}")
    print(f"1号余热锅炉: 修正后平均 ratio = {avg_corrected_hrsg1:.6f}")
    print()
    print(f"2号余热锅炉: 样本点数 = {table['hrsg2_ratio'].notna().sum()}")
    print(f"2号余热锅炉: 原始平均 ratio = {avg_raw_hrsg2:.6f}")
    print(f"2号余热锅炉: 烟气流量修正系数 = {coeff_hrsg2:.6f}")
    print(f"2号余热锅炉: 修正后平均 ratio = {avg_corrected_hrsg2:.6f}")
    print()
    print("抽样 idx:")
    print(table["time_step"].tolist())

    plot_ratios(table, coeff_hrsg1, coeff_hrsg2)


if __name__ == "__main__":
    main()
