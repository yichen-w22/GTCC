import pandas as pd
import os
import fastavro
from pathlib import Path

KEY = "jyrd"
CODE = "JYRD_11HAG03CM001BT03XQ01"

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = {
    "test": Path(r"D:/avrodata"),
    "jqrd": Path(r"E:\beijing_energy\jqrd\taosdump.3494408741406\data0"),
    "jyrd": Path(r"F:/jyrd/taosdump.3514126612530\data0")
}


print(f"开始读取文件")
code = CODE
code = code.replace(".", "_")
# code = code.replace(":", "_")

# 将avro_mapping读取为df
avro_mapping_path = BASE_DIR / Path(KEY) / Path("mapping") / Path(f"avro_mapping_{KEY}.csv")
avro_mapping = pd.read_csv(avro_mapping_path)

# 4) 建立映射表，便于快速查询
tb_to_file = dict(zip(avro_mapping["datacode/测点编码"], 
                      zip(avro_mapping["name/测点名称"], avro_mapping["file_name"])))

param, file_name = tb_to_file.get(code)

avro_path = Path(RAW_DATA_PATH[KEY]) / Path(file_name)
with avro_path.open("rb") as f:
    reader = fastavro.reader(f)
    records = ((r["ts"], r["col0"]) for r in reader)
    df = pd.DataFrame.from_records(records, columns=["ts", "col0"])
    
# 1) 时间戳转为 datetime（UTC）
df["ts"] = pd.to_datetime(df["ts"], unit="ns", utc=True)

# 2) 转为上海时区
df["ts"] = df["ts"].dt.tz_convert("Asia/Shanghai")

# 3) 设为时间索引
df = df.set_index("ts")

# 4) 再进行 1 分钟重采样
df = df.resample("1min").mean()

code = code.replace(":", "_")
df.to_csv(BASE_DIR / Path(f"temp") / Path(f"{param}+{code}.csv"))

print(f"已读取文件{param}+{code}")