# 查看未经筛选的原始数据

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent.parent

p = BASE_DIR / Path(r"datareader_new\jqrd\outcome\jqrd燃机1_1min.pkl")
df_raw = pd.read_pickle(p)
cols = pd.Series(df_raw.columns)
df_raw.columns = cols + cols.groupby(cols).cumcount().replace(0, "").astype(str).radd(".").replace(".", "")

df = pd.DataFrame(index=df_raw.index)

# timestamp
df["timestamp"] = df_raw.index

# "TEMP AMBIENT AIR",
df["TEMP AMBIENT AIR"] = df_raw["TEMP AMBIENT AIR"]

# REL HUMDITY AMB AIR
df["REL HUMDITY AMB AIR"] = df_raw["REL HUMDITY AMB AIR"]

# comp_inlet_T_K
cols_comp_inlet_T_K = [
    "GT COMPR INLET AIR TEMP",
    "TEMP COMPR INLET GT1",
    "TEMP COMPR INLET",
    "TEMP COMPR INLET.1",
    "T COMPR INLET AVE",
    "TEMP COMPR INLET.2",
    "TEMP COMPR INLET.3",
    "TEMP COMPR INLET.4"
]
df["comp_inlet_T_K"] = df_raw[cols_comp_inlet_T_K].mean(axis=1) + 273.15  # 摄氏度转开尔文

# comp_inlet_P_MPa
df["comp_inlet_P_MPa"] = df_raw["PRES U/STR COMPR"] * 0.0001

# comp_exit_T_K
cols_comp_exit_T_K = [
    "TEMP COMPR OUTLET",
    "TEMP COMPR OUTLET.1",
    "TEMP COMPR OUTLET.2",
]
df["comp_exit_T_K"] = df_raw[cols_comp_exit_T_K].mean(axis=1) + 273.15

# comp_exit_P_MPa
cols_comp_exit_P_MPa = [
    "PRES COMPR OUTLET",
    "PRES COMPR OUTLET.1",
    "PRES COMPR OUTLET.2",
]
df["comp_exit_P_MPa"] = df_raw[cols_comp_exit_P_MPa].mean(axis=1)

# IGV_deg
df["IGV_deg"] = df_raw["ACTUAL POSN IGV"]

# shaft_rpm
cols_rps = ["TURBINE SPEED", "TURBINE SPEED.1"]
col_rpm = "TURBINE SPEED.2"

df["shaft_rpm"] = (
    df_raw[cols_rps].sum(axis=1) * 60.0
    + df_raw[col_rpm]
) / 3

# turbine_exhaust_T_K
cols1 = df_raw.columns[df_raw.columns.str.contains("TEMP TURB OUTLET", na=False)]
cols2 = df_raw.columns[df_raw.columns.str.contains("TEMP EXHAUST DUCT", na=False)]
cols_turbine_exhaust_T_K = cols1.tolist() + cols2.tolist()
df["turbine_exhaust_T_K"] = df_raw[cols_turbine_exhaust_T_K].mean(axis=1) + 273.15

# turbine_exit_P_MPa（表压）
cols_turbine_exit_P_MPa = [
    "进口烟压",
    "进口烟压2",
    "进口烟压3",
    "进口烟压4",
]
df["turbine_exit_P_kPa"] = df_raw[cols_turbine_exit_P_MPa].mean(axis=1)

# unit_power_MW
df["unit_power_MW"] = df_raw["APPARENT POWER"]

# turbine_exhaust_flow_kg_s（烟气体积流量）
df["turbine_exhaust_flow"] = df_raw["余热锅炉出口烟气流量"]

# 计算等熵效率
gamma = 1.385

T1 = df["comp_inlet_T_K"]
T2 = df["comp_exit_T_K"]
P1 = df["comp_inlet_P_MPa"]
P2 = df["comp_exit_P_MPa"]

PR = P2 / P1
T2s = T1 * (PR ** ((gamma - 1.0) / gamma))

df["eta_isentropic"] = (T2s - T1) / (T2 - T1)

cols1 = [
    "TEMP AMBIENT AIR",
    "REL HUMDITY AMB AIR",
    "comp_inlet_T_K",
    "comp_inlet_P_MPa",
    "IGV_deg"
]

df[cols1] = df[cols1].apply(
    lambda s: s.interpolate(
        method="linear",
        limit=9,
        limit_area="inside"
    )
)

df.to_csv(r"data_processing\for_degradation\outcome\jqrd_compressor_degradation0.csv", encoding="utf-8-sig", index=False)