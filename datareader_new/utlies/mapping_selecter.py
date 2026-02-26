import pandas as pd

df = pd.read_csv(r"jqrd/mapping/jqrd.csv", encoding="gbk")
print(df.shape)
df = df[df["data_type/数据类型"] == "AI"]
print(df.shape)

# keywords = ["电压", "电流", "线圈", "发电机定子铁心温度", "热氢"
#             , "轴承",  "暖", "氨", "定子", "励磁", "变压器", "油温", "室"
#             , "磁", "设备", "滤", "渗透", "盐", "备用"
#             , "导电率", "润滑", "CURRENT", "VOLTAGE", "绝缘"
#             , "SHAFT", "BRG", "OIL", "SYSTEM", "BEARING", "ACCL", "STR", "GEN", "FAN"
#             , "WNDG", "%", "累计", "oil", "编码", "/", "STARTS", "EQUIV", "FREQ", "天然气压缩机"
#             , "天然气压缩机", "增压机", "瞬时", "AGC", "无功", "频", "油", "冷氢", "冷却器"
#             , "瓦", "电导", "ADM", "电子", "色谱分析", "酸", "碱", "计量", "防"
#             , "装置", "柴油机", "鼓风机", "绕组", "振动", "FILTER"
#             , "厂用", "储气罐", "风冷", "累积", " VOLT", " CUR", "RVSE", "TANK", " SP"
#             , "STATOR", "LOOP", "CTRL", "CRITERION", "START", "TSC"
#             , "干燥机", "空压机", "SETP", "EXC", "粉尘", "信号", "WRKG PWR", "SHIELD", "MFLD"
#             , "EVENTS", "HOURS", "MIN", "MAX", "PILOT", "", "", ""]
# # , "", "", "", "", "", "", "", "", ""

# keywords.extend(["汽机", "氢纯度", "冷却水"
#                  , "气耗率", "蒸汽", "低压缸", "中压缸", "高压缸", "HP", "IP", "LP", "凝汽器"
#                  , "液位", "CONDENSER", "再热", "汽封", "抽气", "循环", "水", "汽"
#                  , "凝结水", "疏水", "排汽", "锅炉", "旁路", "门", "烟", "调节阀", "热网"
#                  , "省煤器", "锅", "除氧器", "过热器", "STEAM", "REHEAT"
#                  , "蒸发器", "反应器", "FLUGAS", ""
#                  ])


# pattern = "|".join(keywords)

# df = df[~df["name/测点名称"].str.contains(pattern, na=False)]
# print(df.shape)

df = df[df["datacode/测点编码"].str.contains("JQRD_11")]
df = df[~df["datacode/测点编码"].str.contains("OUT")]

df.to_csv("temp/1.csv", encoding="utf-8-sig", index=False)