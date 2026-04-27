import pandas as pd
import matplotlib.pyplot as plt


ROW_SLICE = slice(100, 10000)
df1 = pd.read_csv(r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\energy_analysis\GT_model\result\gt1_results_wide.csv").iloc[ROW_SLICE].copy()
df2 = pd.read_csv(r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\energy_analysis\GT_model\result\gt2_results_wide.csv").iloc[ROW_SLICE].copy()



def mass_correction(df):
    df["power_residual"] = df["net_power"] - df["actual_power"]
    df["mass_residual"] = df["power_residual"] / (df["state_4_h"] - df["state_1_h"])
    df["mass_correction_ratio"] = df["mass_residual"] / df["state_1_m_dot"]
    df["corrected_mass"] = df["state_1_m_dot"] + df["mass_residual"]
    df["air_fuel_ratio"] = df["state_1_m_dot"] / df["fuel_m_dot"]
    return df

df1_corrected = mass_correction(df1)
df2_corrected = mass_correction(df2)


prop1 = "mass_residual" 
prop2 = "air_fuel_ratio"
prop3 = "air_fuel_ratio"
prop4 = "actual_power"
prop5 = "mass_correction_ratio"
props = [prop1, prop2, prop3, prop4, prop5]
prop = len(props)
ROLLING_WINDOW = 50

plt.figure(figsize=(18, 9))
plt.subplot(prop, 2, 1)
plt.grid()
plt.plot(df1_corrected[prop1].rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT1_{prop1}")
plt.plot((220 - df1_corrected[prop2] * 5).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT1_{prop2}")
# plt.plot((-35 + df1_corrected[prop4] * 0.0000003).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT1_{prop4}")
# plt.plot((df1_corrected[prop5] * 400).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT1_{prop5}")
plt.legend(framealpha=0.2)    
plt.subplot(prop, 2, 2)
plt.grid()
plt.plot(df2_corrected[prop1].rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT2_{prop1}")
plt.plot((220 - df2_corrected[prop2] * 5).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT2_{prop2}")
# plt.plot((-55 + df2_corrected[prop4] * 0.0000003).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT2_{prop4}")
# plt.plot((df2_corrected[prop5] * 400).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT2_{prop5}")
plt.legend(framealpha=0.2)

plt.subplot(prop, 2, 3)
plt.grid()
plt.plot(df1_corrected[prop5].rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT1_{prop5}")
plt.plot((0.3-df1_corrected[prop2]*0.008).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT1_{prop2}")
# plt.plot((-35 + df1_corrected[prop4] * 0.0000003).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT1_{prop4}")
# plt.plot((df1_corrected[prop5] * 400).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT1_{prop5}")
plt.legend(framealpha=0.2)    
plt.subplot(prop, 2, 4)
plt.grid()
plt.plot(df2_corrected[prop5].rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT2_{prop5}")
plt.plot((0.3-df2_corrected[prop2]*0.008).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT2_{prop2}")
# plt.plot((-55 + df2_corrected[prop4] * 0.0000003).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT2_{prop4}")
# plt.plot((df2_corrected[prop5] * 400).rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label=f"GT2_{prop5}")
plt.legend(framealpha=0.2)
for i in range(2, prop):
    plt.subplot(prop, 2, 2 * i + 1)
    plt.plot(df1_corrected[props[i]].rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label="GT1_" + props[i])
    plt.grid()
    plt.legend()
    plt.subplot(prop, 2, 2 * i + 2)
    plt.plot(df2_corrected[props[i]].rolling(window=ROLLING_WINDOW, min_periods=1).mean(), label="GT2_" + props[i])
    plt.grid()
    plt.legend()
plt.show()

