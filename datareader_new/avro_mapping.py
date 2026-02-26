from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import fastavro
import pandas as pd

from config import RAW_DATA_PATH, KEY, MAPPING


def get_param_from_path(file_path: Path):
    """
    从单个 avro 文件中提取 tbname 和 file_name
    返回: (tbname, file_name) 或 None
    """
    with file_path.open("rb") as f:
        reader = fastavro.reader(f)
        try:
            first_record = next(reader)
        except StopIteration:
            # 空文件，跳过
            return None

        tbname = first_record.get("tbname")
        if tbname is None:
            # 没有 tbname 字段，跳过
            return None
        tbname = tbname.replace(".", "_")
        return tbname, file_path.name


def build_avro_mapping():
    
    out_path = Path(KEY) / "mapping" / f"avro_mapping_{KEY}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    
    if MAPPING == False:
        return
    
    """
    为指定电厂构建 avro 映射表:
    tbname → file_name
    并保存为 points/avro_mapping_{plant_code}.csv
    """
    root = RAW_DATA_PATH[KEY]
    files = list(root.glob("*.avro"))

    print(f"[{KEY}] 在 {root} 下共发现 {len(files)} 个 avro 文件，开始扫描…")

    records = []

    # 多进程并行读取 50000 个文件
    with ProcessPoolExecutor() as executor:
        from tqdm import tqdm

        results = executor.map(get_param_from_path, files, chunksize=100)

        for result in tqdm(results, total=len(files), desc=f"Building avro mapping for {KEY}"):
            if result is None:
                continue
            tbname, fname = result
            records.append((tbname, fname))

    df_map = pd.DataFrame(records, columns=["datacode/测点编码", "file_name"])
    
    mapping_csv = Path(KEY) / Path("mapping") / Path(f"{KEY}.csv")
    df_mapping_csv = pd.read_csv(mapping_csv, encoding="gbk")[["datacode/测点编码", "name/测点名称"]]
    df_mapping_csv["datacode/测点编码"] = df_mapping_csv["datacode/测点编码"].str.replace(".", "_")
    
    df_map = df_map.merge(df_mapping_csv, on="datacode/测点编码", how="left")
    df_map.insert(0, "name/测点名称", df_map.pop("name/测点名称"))

    df_map.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[{KEY}] 映射构建完成，共 {len(df_map)} 条记录")
    print("已保存到：", out_path)


if __name__ == "__main__":
    build_avro_mapping()
    

