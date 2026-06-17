"""一键运行所有绘图脚本，生成全部论文图片。"""

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

MODULES = [
    "ch3_gt_power_comparison",
    "ch3_gt_component_efficiency",
    "ch3_gt_energy_distribution",
    "ch3_gt_energy_share",
    "ch4_fluegas_energy_distribution",
    "ch4_fluegas_energy_share",
    "ch4_waterside_heat_distribution",
    "ch4_waterside_heat_share",
    "ch4_st_power_comparison",
    "ch4_st_cylinder_distribution",
    "ch4_st_cylinder_share",
    "ch4_st_cylinder_efficiency",
    "ch5_correlation_matrix",
    "ch5_efficiency_factor_plots",
]


def main() -> None:
    for name in MODULES:
        print(f"\n{'='*60}")
        print(f"运行 {name}")
        print(f"{'='*60}")
        mod = importlib.import_module(name)
        mod.main()
    print(f"\n{'='*60}")
    print("全部绘图完成。")


if __name__ == "__main__":
    main()
