import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\continuous_data_10min.csv"

POWER_COL_1 = "\u71c3\u673a\u51fa\u529b_1"
POWER_COL_2 = "\u71c3\u673a\u51fa\u529b_2"
FUEL_MASS_FLOW_COL_1 = "\u71c3\u6599\u8d28\u91cf\u6d41\u91cf_1"
FUEL_MASS_FLOW_COL_2 = "\u71c3\u6599\u8d28\u91cf\u6d41\u91cf_2"
POWER_LABEL = "\u71c3\u673a\u51fa\u529b"
FUEL_MASS_FLOW_LABEL = "\u71c3\u6599\u8d28\u91cf\u6d41\u91cf"

df = pd.read_csv(DATA_PATH)


def plot_linear_fit(ax, x, y, label):
    data = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    k, b = np.polyfit(data["x"], data["y"], 1)

    ax.scatter(data["x"], data["y"], label=label)
    ax.plot(
        data["x"],
        k * data["x"] + b,
        color="red",
        label=f"{label} fit: y = {k:.6g}x + {b:.6g}",
    )
    ax.set_xlabel(POWER_LABEL)
    ax.set_ylabel(FUEL_MASS_FLOW_LABEL)
    ax.legend()

    print(f"{label} fitting line: y = {k:.10g}x + {b:.10g}")


plt.figure(figsize=(18, 9))

ax1 = plt.subplot(1, 2, 1)
plot_linear_fit(ax1, df[POWER_COL_1], df[FUEL_MASS_FLOW_COL_1], "GT_1")

ax2 = plt.subplot(1, 2, 2)
plot_linear_fit(ax2, df[POWER_COL_2], df[FUEL_MASS_FLOW_COL_2], "GT_2")

plt.tight_layout()
plt.show()
