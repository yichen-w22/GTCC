import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
p = BASE_DIR / Path(r"datareader_new\jqrd\outcome\jqrd燃机1_1min.pkl")

df_raw = pd.read_pickle(p)
cols = pd.Series(df_raw.columns)
df_raw.columns = cols + cols.groupby(cols).cumcount().replace(0, "").astype(str).radd(".").replace(".", "")

df = pd.DataFrame(index=df_raw.index)

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
df["comp_inlet_T_K"] = df_raw[cols_comp_inlet_T_K].mean(axis=1) + 273.15
df["comp_inlet_P_MPa"] = df_raw["PRES U/STR COMPR"] * 0.0001

cols_comp_exit_T_K = ["TEMP COMPR OUTLET", "TEMP COMPR OUTLET.1", "TEMP COMPR OUTLET.2"]
df["comp_exit_T_K"] = df_raw[cols_comp_exit_T_K].mean(axis=1) + 273.15

cols_comp_exit_P_MPa = ["PRES COMPR OUTLET", "PRES COMPR OUTLET.1", "PRES COMPR OUTLET.2"]
df["comp_exit_P_MPa"] = df_raw[cols_comp_exit_P_MPa].mean(axis=1)

df["IGV_deg"] = df_raw["ACTUAL POSN IGV"]

cols_rps = ["TURBINE SPEED", "TURBINE SPEED.1"]
col_rpm = "TURBINE SPEED.2"
df["shaft_rpm"] = (df_raw[cols_rps].sum(axis=1) * 60.0 + df_raw[col_rpm]) / 3

df["unit_power_MW"] = df_raw["APPARENT POWER"]

df = df.where(df["shaft_rpm"] > 2950)

cols_interp2 = ["comp_inlet_T_K", "comp_inlet_P_MPa"]
df[cols_interp2] = df[cols_interp2].apply(
    lambda s: s.interpolate(method="linear", limit=30, limit_area="inside")
)

gamma = 1.385
T1 = df["comp_inlet_T_K"]
T2 = df["comp_exit_T_K"]
P1 = df["comp_inlet_P_MPa"]
P2 = df["comp_exit_P_MPa"]

PR = P2 / P1
T2s = T1 * (PR ** ((gamma - 1.0) / gamma))
df["eta_isentropic"] = (T2s - T1) / (T2 - T1)

# ====== 找“所有列同时无 NaN”的最长连续区间（按 df.index 连续） ======
valid = df.notna().all(axis=1)

grp = valid.ne(valid.shift()).cumsum()

segments = valid.groupby(grp).agg(valid="first", length="size")
segments_valid = segments[segments["valid"]]

if len(segments_valid) == 0:
    start_index = end_index = None
    print("没有任何一个区间能做到：所有列同时无 NaN")
else:
    gid = segments_valid["length"].idxmax()
    idx = df.index[grp == gid]
    start_index = idx[0]
    end_index = idx[-1]

    print("最长无空值区间：")
    print("start_index =", start_index)
    print("end_index   =", end_index)
    print("length      =", len(idx))

# （可选）切出最长段
df_longest = df.loc[start_index:end_index] if start_index is not None else df.iloc[0:0]

df_longest.to_csv(r"temp/temp_longest_allcols.csv", encoding="utf-8-sig")