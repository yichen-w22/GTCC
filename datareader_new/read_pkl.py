import pandas as pd
from pathlib import Path

p = Path(r"D:\清华\毕业设计\test\datareader_new\jqrd\outcome\jqrd燃机1_1min.pkl")
df = pd.read_pickle(p)
print(df.columns.value_counts())

for col in df.columns:
    print(col)

# df = df.resample("10min").mean()
# # print(df["燃机前置模块天然气进气流量"])

# # df = df.resample("10min").mean()

# df.to_csv(r"temp\temp.csv", encoding="utf-8-sig")