import pandas as pd
import json
from pathlib import Path

with open('jyrd/need/need.json', 'r', encoding='utf-8') as f:
    need = json.load(f)

# 初始化参数
key = 'lt9'  # whose need

columns = need[key]
# 对list的每一个元素，若存在.，则替换为_
columns = [col.replace('.', '_') for col in columns]

# 把 columns 放进 DataFrame
df_columns = pd.DataFrame(
    {"datacode/测点编码": columns}
)

df_columns.to_excel(r"jyrd/need/need_lt9.xlsx", index=False)