from pathlib import Path
import pandas as pd

# 选取电厂
KEY = "jqrd"

#选取开始和结束时间
# start_time = pd.Timestamp('2025-6-1')
# end_time = pd.Timestamp('2025-8-1')
# TIMERANGE = pd.date_range(
#     start=start_time, end=end_time, freq='1min')
TIMERANGE = "ALL"

# 选取采样频率
FREQUENCY = "1min"

# 选择是否生成avro_mapping
# MAPPING = True
MAPPING = True


# 电厂代码列表
PLANTS = ["test", "jqrd", "jyrd"]

# 数据路径配置
RAW_DATA_PATH = {
    "test": Path(r"D:/avrodata"),
    "jqrd": Path(r"C:\MyFolder\Projects\GTCC\Data\jqrd\part2\taosdump.3540568337482\data0-657F0827"),
    "jyrd": Path(r"D:\beijing_energy\jyrd\taosdump.3514126612530\data0")
}

