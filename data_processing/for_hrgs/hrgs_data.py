import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


BASE_DIR = Path(__file__).resolve().parent.parent.parent
p = BASE_DIR / Path(r"datareader_new\jyrd\outcome\jyrd高压余热锅炉_1min.pkl")

df_raw = pd.read_pickle(p)

cols = pd.Series(df_raw.columns)
df_raw.columns = cols.mask(cols.isna() | (cols == ""), cols.shift())

cols = pd.Series(df_raw.columns)
df_raw.columns = cols + cols.groupby(cols).cumcount().replace(0, "").astype(str).radd(".").replace(".", "")

df_raw = df_raw.replace(0, np.nan)

# df_raw = df_raw[df_raw["#1炉烟囱出口烟气流量"] > 100000]

df = pd.DataFrame(index=df_raw.index)
df["timestamp"] = df_raw.index

df["高压过热器1入口烟温"] = df_raw[[f"#1炉高压过热器1入口烟温{i}" for i in range(1,7)]].mean(axis=1)

df["高压蒸发器入口烟温"] = df_raw[[f"#1炉高压蒸发器入口烟温{i}" for i in range(1,7)]].mean(axis=1)

df["高压省煤器3入口烟温"] = df_raw[[f"#1炉高压省煤器3入口烟温{i}" for i in range(1,7)]].mean(axis=1)

df["中压过热器入口烟温"] = df_raw[[f"#1炉中压过热器入口烟温{i}" for i in range(1,7)]].mean(axis=1)

df["烟道入口烟气压力"] = df_raw["#1炉烟道入口烟气压力"]

df["烟道出口烟气压力"] = df_raw["#1炉出口烟气压力"]
df["烟囱出口烟气压力"] = df_raw["#1炉烟囱出口烟气压力"]

df["烟囱出口烟气流量"] = df_raw["#1炉烟囱出口烟气流量"]

df["燃机烟气流速"] = df_raw["#1燃机烟气流速"]

#1炉#1高压给水泵出口流量
df["#1高压给水泵出口流量"] = df_raw["#1炉#1高压给水泵出口流量"]

#1炉#1高压给水泵入口压力
df["#1高压给水泵入口压力"] = df_raw["#1炉#1高压给水泵入口压力"]

#1炉#1高压给水泵出口压力
df["#1高压给水泵出口压力"] = df_raw["#1炉#1高压给水泵出口压力"]

#1炉#2高压给水泵出口流量
df["#2高压给水泵出口流量"] = df_raw["#1炉#2高压给水泵出口流量"]

#1炉#2高压给水泵入口压力
df["#2高压给水泵入口压力"] = df_raw["#1炉#2高压给水泵入口压力"]

#1炉#2高压给水泵出口压力
df["#2高压给水泵出口压力"] = df_raw["#1炉#2高压给水泵出口压力"]

高压给水流量 = [
    "#1炉高压给水流量A",
    "#1炉高压给水流量A.1",
    "#1炉高压给水流量B",
    "#1炉高压给水流量B.1",
    "#1炉高压给水流量C",
    "#1炉高压给水流量C.1",
    "#1锅炉高压给水流量"
]
df["高压给水流量"] = df_raw[高压给水流量].mean(axis=1)

#1炉高压给水温度
df["高压给水温度"] = df_raw["#1炉高压给水温度"]

#1炉高压给水母管压力
df["高压给水母管压力"] = df_raw["#1炉高压给水母管压力"]

#1炉高压省煤器1入口压力
df["高压省煤器1入口压力"] = df_raw["#1炉高压省煤器1入口压力"]

#1炉高压省煤器1出口压力
df["高压省煤器1出口压力"] = df_raw["#1炉高压省煤器1出口压力"]

高压省煤器2出口给水温度 = [
    "#1炉高压省煤器2出口给水温度A",
    "#1炉高压省煤器2出口给水温度B",
    "#1炉高压省煤器2出口给水温度C",
    "#1炉高压省煤器2出口给水温度D",
]
df["高压省煤器2出口给水温度"] = df_raw[高压省煤器2出口给水温度].mean(axis=1)

高压省煤器3出口给水温度 = [
    "#1炉高压省煤器3出口给水温度A",
    "#1炉高压省煤器3出口给水温度B",
    "#1炉高压省煤器3出口给水温度C",
    "#1炉高压省煤器3出口给水温度D",
]
df["高压省煤器3出口给水温度"] = df_raw[高压省煤器3出口给水温度].mean(axis=1)


高压汽包水位 = [
    "#1炉高压汽包水位A.1",
    "#1炉高压汽包水位B.1",
    "#1炉高压汽包水位C.1",
    "#1炉高压汽包水位"
]
df["高压汽包水位"] = df_raw[高压汽包水位].mean(axis=1)

锅炉高压汽包压力 = [
    "#1炉高压汽包压力A",
    "#1炉高压汽包压力B",
    "#1炉高压汽包压力C",
    "#1锅炉高压汽包压力",
]
df["锅炉高压汽包压力"] = df_raw[锅炉高压汽包压力].mean(axis=1)

高压汽包上壁温 = [
    "#1炉高压汽包上壁温A",
    "#1炉高压汽包上壁温B",
    "#1炉高压汽包上壁温C",
]
df["高压汽包上壁温"] = df_raw[高压汽包上壁温].mean(axis=1)

高压汽包下壁温 = [
    "#1炉高压汽包下壁温A",
    "#1炉高压汽包下壁温B",
    "#1炉高压汽包下壁温C",
]
df["高压汽包下壁温"] = df_raw[高压汽包下壁温].mean(axis=1)

#1炉高压过热蒸汽减温水流量
df["高压过热蒸汽减温水流量"] = df_raw["#1炉高压过热蒸汽减温水流量"]

#1炉高压过热汽减温器入口蒸汽温度
df["高压过热汽减温器入口蒸汽温度"] = df_raw["#1炉高压过热汽减温器入口蒸汽温度"]

高压过热汽减温器出口蒸汽温度 = [
    "#1炉高压过热汽减温器出口蒸汽温度A",
    "#1炉高压过热汽减温器出口蒸汽温度B",
    "#1炉高过减温器出口蒸汽温度",
]
df["高压过热汽减温器出口蒸汽温度"] = df_raw[高压过热汽减温器出口蒸汽温度].mean(axis=1)

#1炉高压过热蒸汽减温器出口压力
df["高压过热蒸汽减温器出口压力"] = df_raw["#1炉高压过热蒸汽减温器出口压力"]

#1炉高压过热汽减温器出口疏水温度
df["高压过热汽减温器出口疏水温度"] = df_raw["#1炉高压过热汽减温器出口疏水温度"]



#1炉高压主蒸汽流量
df["高压主蒸汽流量"] = df_raw["#1炉高压主蒸汽流量"]

高压主汽压力 = [
    "#1炉高压主蒸汽压力A",
    "#1炉高压主蒸汽压力B",
    "#1炉高压主蒸汽压力C",
    "#1锅炉高压主汽压力"
]
df["高压主汽压力"] = df_raw[高压主汽压力].mean(axis=1)

高压主蒸汽温度 = [
    "#1炉高压主蒸汽温度A",
    "#1炉高压主蒸汽温度B",
    "#1炉高压主蒸汽温度C",
    "#1锅炉高压主汽温度"
]
df["高压主蒸汽温度"] = df_raw[高压主蒸汽温度].mean(axis=1)

df.to_csv(
    r"data_processing\for_hrgs\outcome\jyrd_hrgs_#1.csv",
    encoding="utf-8-sig",
    index=False
)

# cols_temp = [f"#1炉高压省煤器3入口烟温{i}" for i in range(1,7)]

# plt.figure(figsize=(10, 5))

# for c in cols_temp:
#     plt.scatter(df["timestamp"], df_raw[c], s=1, label=c, alpha=0.1)

# plt.xlabel("time")
# plt.ylabel("Temperature")
# plt.title("高压省煤器3入口烟温 各测点随时间变化")
# plt.legend()
# plt.tight_layout()
# plt.show()

# start = "2025-07-01 00:00:00"
# end   = "2025-07-01 00:00:00"

# mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)

# plt.figure(figsize=(10, 5))

# cols_temp = [f"#1炉高压省煤器3入口烟温{i}" for i in range(1,7)]

# for c in cols_temp:
#     plt.plot(df.loc[mask, "timestamp"],
#              df_raw.loc[mask, c],
#              linewidth=0.8,
#              label=c)

# plt.xlabel("time")
# plt.ylabel("Temperature")
# plt.title("")
# plt.legend()
# plt.tight_layout()
# plt.show()