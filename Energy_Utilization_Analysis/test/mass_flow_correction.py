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
    return k, b


def plot_reference_fit(ax, x, y, label):
    data = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    k, b = np.polyfit(data["x"], data["y"], 1)

    ax.scatter(data["x"], data["y"], label=label, color="black", marker="x", s=100, zorder=3)
    ax.plot(
        data["x"],
        k * data["x"] + b,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"{label} fit: y = {k:.6g}x + {b:.6g}",
        zorder=2,
    )
    ax.legend()

    print(f"{label} fitting line: y = {k:.10g}x + {b:.10g}")
    return k, b

plt.figure(figsize=(18, 9))

ax1 = plt.subplot(1, 2, 1)
plot_linear_fit(ax1, df[POWER_COL_1], df[FUEL_MASS_FLOW_COL_1], "GT_1")
x = [277.5e6, 218e6, 195e6, 172e6, 278.25e6, 241.17e6]
y = [16.18010094, 13.14728693, 12.13889537, 11.19161845, 16.09606831, 14.254989779999999]
plot_reference_fit(ax1, x, y, "real data")

ax2 = plt.subplot(1, 2, 2)
plot_linear_fit(ax2, df[POWER_COL_2], df[FUEL_MASS_FLOW_COL_2], "GT_2")
x = [278.30e6, 218.82e6, 195.41e6, 173.20e6, 275.1e6, 242.69e6]
y = [16.01203568, 12.994500330000001, 11.9937481, 11.18397912, 16.042593, 14.39249772]
plot_reference_fit(ax2, x, y, "real data")

plt.tight_layout()
plt.show()
