import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 这里是对燃机出力的修正

def scatter_with_fit(x, y, label, s=15, alpha=0.1):
    """Draw scatter points and their linear fit line."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return None

    k, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = k * x_line + b

    plt.scatter(x, y, s=s, alpha=alpha, label=label)
    # plt.plot(x_line, y_line, linewidth=2, label=f"{label} fit", color=(0,0,0))

    return k, b

df_jqrd = pd.read_csv(r'C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\continuous_data_10min.csv')

df_jyrd = pd.read_csv(r'C:\MyFolder\Projects\GTCC\Test\datareader_new\jyrd\outcome\jyrd燃机_1min.csv')

df_test = pd.read_csv(r'C:\MyFolder\Projects\GTCC\Test\datareader_new\jqrd\outcome\test_1min.csv')



df_jqrd["燃料质量流量_1"] = df_jqrd["燃料质量流量_1"]
df_jqrd["燃料质量流量_2"] = df_jqrd["燃料质量流量_2"]
df_jyrd["Gas Fuel Flow"] = df_jyrd["Gas Fuel Flow"] * 0.45359237
df_jyrd["Gas Fuel Flow.1"] = df_jyrd["Gas Fuel Flow.1"] * 0.45359237
df_jyrd["Turbine Exhaust Mass Flow"] = df_jyrd["Turbine Exhaust Mass Flow"] * 0.45359237
df_jyrd["Turbine Exhaust Mass Flow.1"] = df_jyrd["Turbine Exhaust Mass Flow.1"] * 0.45359237
df_jyrd = df_jyrd[df_jyrd["Gas Fuel Flow"] > 6]
df_jyrd = df_jyrd[df_jyrd["Gas Fuel Flow.1"] > 6]
#  / 0.764 * 3600
# for c in df_jyrd.columns:
#     print(c)

# 由华氏度转为开尔文
df_jyrd["Exhaust Temp MeDXan Corrected By Average"] = (df_jyrd["Exhaust Temp MeDXan Corrected By Average"] - 32) * 5.0/9.0 + 273.15
df_jyrd["Exhaust Temp MeDXan Corrected By Average.1"] = (df_jyrd["Exhaust Temp MeDXan Corrected By Average.1"] - 32) * 5.0/9.0 + 273.15
df_jyrd["GT2实发"] = df_jyrd["GT2实发"] * 1e6
df_jyrd["一拖一总负荷预设P1值"] = df_jyrd["一拖一总负荷预设P1值"] * 1e6
df_jyrd = df_jyrd[df_jyrd["GT2实发"] > 100e6]
df_jyrd = df_jyrd[df_jyrd["一拖一总负荷预设P1值"] > 100e6]
# plt.figure(figsize=(12, 6))
# plt.scatter(df_jyrd["GT2实发"], df_jyrd["Exhaust Temp MeDXan Corrected By Average"], label="jyrd1")
# plt.scatter(df_jyrd["一拖一总负荷预设P1值"], df_jyrd["Exhaust Temp MeDXan Corrected By Average.1"], label="jyrd2")
# plt.scatter(df_jqrd["燃机出力_1"], df_jqrd["透平排气温度_1"], label="jqrd1")
# plt.scatter(df_jqrd["燃机出力_2"], df_jqrd["透平排气温度_2"], label="jqrd2")
# plt.scatter(df_test["APPARENT POWER"], df_test["TEMP EXHAUSTAVER"], label="test1")
# plt.xlim(1e8, 2.5e8)
# plt.ylim(800, 1000)
# x = [277.75e6, 218.38e6, 194.96e6, 172.71e6]
# y = [76243.22, 61942.46, 57221.24, 52753]
# plt.scatter(df_jqrd["燃机出力_1"], df_jqrd["燃料质量流量_1"], label="jqrd1")
# plt.scatter(df_jqrd["燃机出力_1"], df_jqrd["燃料质量流量_2"], label="jqrd1")
# plt.plot(x, y, "x", zorder=10)
# plt.grid()
# plt.legend()
# plt.show()

# scatter_with_fit(df_jyrd["一拖一总负荷预设P1值"], df_jyrd["Gas Fuel Flow"], label="jyrd1")
# scatter_with_fit(df_jyrd["GT2实发"], df_jyrd["Gas Fuel Flow.1"], label="jyrd2")
# scatter_with_fit(df_jqrd["燃机出力_1"], df_jqrd["燃料质量流量_1"], label="jqrd1")
# scatter_with_fit(df_jqrd["燃机出力_1"], df_jqrd["燃料质量流量_2"], label="jqrd2")
# plt.xlabel("燃机出力")
# plt.ylabel("燃料质量流量")
# plt.grid()
# plt.legend()
# plt.show()

# scatter_with_fit(df_jyrd["一拖一总负荷预设P1值"], df_jyrd["Turbine Exhaust Mass Flow"], label="jyrd1")
# scatter_with_fit(df_jyrd["GT2实发"], df_jyrd["Turbine Exhaust Mass Flow.1"], label="jyrd2")
# scatter_with_fit(df_jqrd["燃机出力_1"], df_jqrd["烟气质量流量_1"], label="jqrd1")
scatter_with_fit(df_jqrd["燃机出力_2"], df_jqrd["烟气质量流量_2"], label="jqrd2")
plt.xlabel("燃机出力")
plt.ylabel("烟气质量流量")
plt.grid()
plt.legend()
plt.show()