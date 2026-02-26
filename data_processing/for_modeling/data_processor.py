import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

p = BASE_DIR / Path(r"datareader_new\jqrd\outcome\jqrd燃机1_1min.pkl")
df_raw = pd.read_pickle(p)  

df_raw = df_raw.resample("10min").mean()
df_raw = df_raw.dropna()
df_raw = df_raw.head(1000)

cols = pd.Series(df_raw.columns)
df_raw.columns = cols + cols.groupby(cols).cumcount().replace(0, "").astype(str).radd(".").replace(".", "")

template = pd.read_csv(r"data_processing\imput_template.csv")
df_imput = pd.DataFrame(columns=template.columns)

df_imput = df_imput.reindex(df_raw.index)

# timestamp
df_imput["timestamp"] = df_raw.index

# source_engine
df_imput["source_engine"] = "#1"


# comp_inlet_T_K
cols_comp_inlet_T_K = [
    "TEMP AMBIENT AIR",
    "GT COMPR INLET AIR TEMP",
    "TEMP COMPR INLET GT1",
    "TEMP COMPR INLET",
    "TEMP COMPR INLET.1",
    "T COMPR INLET AVE",
    "TEMP COMPR INLET.2",
    "TEMP COMPR INLET.3",
    "TEMP COMPR INLET.4"
]
df_imput["comp_inlet_T_K"] = df_raw[cols_comp_inlet_T_K].mean(axis=1) + 273.15  # 摄氏度转开尔文

# comp_inlet_P_MPa
df_imput["comp_inlet_P_MPa"] = df_raw["PRES U/STR COMPR"] * 0.0001

# comp_exit_T_K
cols_comp_exit_T_K = [
    "TEMP COMPR OUTLET",
    "TEMP COMPR OUTLET.1",
    "TEMP COMPR OUTLET.2",
]
df_imput["comp_exit_T_K"] = df_raw[cols_comp_exit_T_K].mean(axis=1) + 273.15

# # 用于计算极差，计算逐行极差
# std_series = df_raw[cols_comp_exit_T_K].std(axis=1)

# plt.figure(figsize=(10, 4))
# plt.plot(std_series)
# plt.grid(True)
# plt.show()

# comp_exit_P_MPa
cols_comp_exit_P_MPa = [
    "PRES COMPR OUTLET",
    "PRES COMPR OUTLET.1",
    "PRES COMPR OUTLET.2",
]
df_imput["comp_exit_P_MPa"] = df_raw[cols_comp_exit_P_MPa].mean(axis=1)

# IGV_deg
df_imput["IGV_deg"] = df_raw["ACTUAL POSN IGV"]

# shaft_rpm
cols_rps = ["TURBINE SPEED", "TURBINE SPEED.1"]
col_rpm = "TURBINE SPEED.2"

df_imput["shaft_rpm"] = (
    df_raw[cols_rps].sum(axis=1) * 60.0
    + df_raw[col_rpm]
) / 3

# turbine_exhaust_T_K
# TEMP_TURB_OUTLET
cols1 = df_raw.columns[df_raw.columns.str.contains("TEMP TURB OUTLET", na=False)]

# TEMP_EXHAUST_DUCT
cols2 = df_raw.columns[df_raw.columns.str.contains("TEMP EXHAUST DUCT", na=False)]

cols_turbine_exhaust_T_K = cols1.tolist() + cols2.tolist()
df_imput["turbine_exhaust_T_K"] = df_raw[cols_turbine_exhaust_T_K].mean(axis=1) + 273.15

# turbine_exit_P_MPa
cols_turbine_exit_P_MPa = [
    "进口烟压",
    "进口烟压2",
    "进口烟压3",
    "进口烟压4",
]
df_imput["turbine_exit_P_MPa"] = df_raw[cols_turbine_exit_P_MPa].mean(axis=1) * 0.001 + df_imput["comp_inlet_P_MPa"]

# unit_power_MW
df_imput["unit_power_MW"] = df_raw["APPARENT POWER"]

# ambient_RH
df_imput["ambient_RH"] = df_raw["REL HUMDITY AMB AIR"]

R = 8.314462618
AW = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007}

SPEC = {
    "H2":     ({"H":2},                 2.0159, 120.0),
    "N2":     ({"N":2},                28.0134,   0.0),
    "CO2":    ({"C":1,"O":2},          44.0095,   0.0),
    "CH4":    ({"C":1,"H":4},          16.0425,  50.0),
    "CO":     ({"C":1,"O":1},          28.0101,  10.1),
    "O2":     ({"O":2},                31.9988,   0.0),
    "C2H6":   ({"C":2,"H":6},          30.0690,  47.5),
    "C3H8":   ({"C":3,"H":8},          44.0956,  46.4),
    "iC4H10": ({"C":4,"H":10},         58.1222,  45.7),
    "nC4H10": ({"C":4,"H":10},         58.1222,  45.7),
    "iC5H12": ({"C":5,"H":12},         72.1488,  45.0),
    "nC5H12": ({"C":5,"H":12},         72.1488,  45.0),
    "C6+":    ({"C":6,"H":14},         86.177,   44.6),
}

keys = ["H2","N2","CO2","CH4","CO","O2+Ar","C2H6","C3H8","iC4H10","nC4H10","iC5H12","nC5H12","C6+"]

df_raw[keys] = df_raw[keys] * 0.01

# 近似：O2+Ar 全当 O2
x = df_raw[keys].copy()
x["O2"] = x.pop("O2+Ar")          # 新增一列 O2

Mmix = 0.0
for sp, (_, MW, _) in SPEC.items():
    if sp in x.columns:
        Mmix += x[sp] * MW

# --- 2) 元素质量（以 1 mol 混合气为基准，单位 g） ---
m_elem = {e: 0.0 for e in ["C", "H", "O", "N"]}
for sp, (formula, MW, _) in SPEC.items():
    if sp not in x.columns:
        continue
    for e in m_elem:
        m_elem[e] += x[sp] * formula.get(e, 0) * AW[e]

# --- 3) 元素质量分数 ---
df_imput["fuel_C_mass_frac"] = m_elem["C"] / Mmix
df_imput["fuel_H_mass_frac"] = m_elem["H"] / Mmix
df_imput["fuel_O_mass_frac"] = m_elem["O"] / Mmix
df_imput["fuel_N_mass_frac"] = m_elem["N"] / Mmix

# --- 4) LHV（MJ/m3 -> J/kg）---
# lhv_num = 0.0
# for sp, (_, MW, LHV) in SPEC.items():
#     if sp in x.columns:
#         lhv_num += x[sp] * MW * LHV  # (g/mol)*MJ/kg
# df_imput["LHV_J_per_kg"] = (lhv_num / Mmix) * 1e6

df_imput["LHV_J_per_kg"] = df_raw["低热值"] / 0.737 * 1e6

# turbine_exhaust_flow_kg_s
rho_std = 1.293  # kg/Nm3
df_imput["turbine_exhaust_flow_kg_s"] = (
    df_raw["余热锅炉出口烟气流量"] * rho_std / 3600
)

# fuel_mass_flow_kg_s
P_std = 101325.0     # Pa
T_std = 273.15       # K

df_imput["fuel_rho_std_kg_m3"] = P_std * (Mmix / 1000.0) / (R * T_std)
df_imput["fuel_mass_flow_kg_s"] = (
    df_raw["1#燃气流量"] * df_imput["fuel_rho_std_kg_m3"] / 3600.0
)

# air_inlet_flow_kg_s
df_imput["air_inlet_flow_kg_s"] = df_imput["turbine_exhaust_flow_kg_s"] - df_imput["fuel_mass_flow_kg_s"]

# fuel_T_K
df_imput["fuel_T_K"] = df_raw["TEMP U/STR NG ESV"] + 273.15

# df_imput = df_imput.dropna()

# 取前10行
df_imput_head = df_imput.head(10)
df_imput_head.to_csv(r"data_processing\outcome\jqrd燃机1_10min.csv", encoding="utf-8-sig", index=False)
