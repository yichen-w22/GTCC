import pandas as pd
import os
import fastavro
from pathlib import Path

avro_path = Path(r'E:\\jqrd\\taosdump.3494408741406\\data0\\d_qjny_jqrd.3494410899958.56788.avro')
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
df = df.resample("10min").mean()

df.to_csv(Path(f"temp") / Path(f"1.csv"))

