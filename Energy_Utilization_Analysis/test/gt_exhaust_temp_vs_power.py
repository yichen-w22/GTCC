"""透平排气温度 vs 燃机实际出力 散点图"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

POWER_LOWER_MW = 100.0
POWER_UPPER_MW = 300.0


def main() -> None:
    metrics = pd.read_csv(PLANT_METRICS_PATH)
    raw = pd.read_csv(RAW_DATA_PATH)
    idx = metrics["idx"].astype(int)

    configs = [
        (1, "GT1.燃机实际功", "透平排气温度_1"),
        (2, "GT2.燃机实际功", "透平排气温度_2"),
    ]

    for unit, power_col, temp_col in configs:
        run_col = f"GT{unit}.是否运行"
        running = metrics[run_col].astype(str).str.upper().isin(["TRUE", "1"])

        power_mw = metrics[power_col] / 1e6
        temp = raw.loc[idx, temp_col].reset_index(drop=True)
        if temp.median() > 200:
            temp = temp - 273.15

        valid = (
            running
            & np.isfinite(power_mw)
            & np.isfinite(temp)
            & power_mw.between(POWER_LOWER_MW, POWER_UPPER_MW)
            & (temp > 300)
            & (temp < 700)
        )

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(
            power_mw[valid], temp[valid],
            s=4, alpha=0.35, edgecolors="none",
        )
        ax.set_xlabel("燃机实际出力 / MW")
        ax.set_ylabel("透平排气温度 / ℃")
        ax.set_xlim(POWER_LOWER_MW, POWER_UPPER_MW)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
