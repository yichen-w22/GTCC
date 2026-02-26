import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import fastavro

from config import RAW_DATA_PATH, KEY, TIMERANGE, FREQUENCY


# =========================
# 固定：缓存频率（pkl_data 永远 1min）
# =========================
CACHE_FREQ = "1min"

# =========================
# 固定：工程根目录 = datareader_new
# get_data.py 位于 datareader_new/ 下
# =========================
BASE_DIR = Path(__file__).resolve().parent

# KEY 目录结构：datareader_new/<KEY>/{mapping,need,outcome,pkl_data}
KEY_DIR = BASE_DIR / KEY
MAPPING_DIR = KEY_DIR / "mapping"
NEED_DIR = KEY_DIR / "need"
OUT_DIR = KEY_DIR / "outcome"
PKL_DIR = KEY_DIR / "pkl_data"


# =========================
# 1) 单文件：avro -> resample(1min) -> pkl cache
# =========================
def avro_file_to_resampled_pkl(avro_path: Path, out_pkl: Path, freq: str) -> dict:
    """
    读取单个 avro -> DataFrame(ts,col0) -> 以 ts 为索引重采样 -> 保存为 pkl
    返回 meta 用于监控统计
    """
    t0 = time.perf_counter()
    size = avro_path.stat().st_size

    with avro_path.open("rb") as f:
        reader = fastavro.reader(f)
        records = ((r.get("ts"), r.get("col0")) for r in reader)
        df = pd.DataFrame.from_records(records, columns=["ts", "col0"])

    # 清洗
    df = df.dropna(subset=["ts", "col0"])
    if df.empty:
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["col0"], index=pd.DatetimeIndex([], name="ts")).to_pickle(out_pkl)
        return {"secs": time.perf_counter() - t0, "bytes": size, "rows": 0, "out": str(out_pkl)}

    # ts ns -> datetime index
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ns", errors="coerce")
    df = df.dropna(subset=["ts"])
    df = df.set_index("ts").sort_index()

    # 缓存：固定 1min
    df = df[["col0"]].resample(freq).mean()

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_pkl)

    return {"secs": time.perf_counter() - t0, "bytes": size, "rows": len(df), "out": str(out_pkl)}


def build_persistent_cache_for_list(
    list_found: list[str],
    cache_dir: Path,
    freq: str = CACHE_FREQ,
    max_workers: int = 2,
) -> list[Path]:
    """
    对 list_found 中的 avro 文件生成/更新【长期缓存】pkl（已存在且比 avro 新则跳过）
    返回对应的 pkl 路径列表（按实际存在的返回）
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for filename in list_found:
        avro_path = RAW_DATA_PATH[KEY] / filename
        if not avro_path.exists():
            print(f"[WARN] Avro 不存在：{avro_path}")
            continue

        out_pkl = cache_dir / f"{filename}.{freq}.pkl"

        # 缓存命中：pkl 存在且比 avro 新（按 mtime）
        if out_pkl.exists() and out_pkl.stat().st_mtime >= avro_path.stat().st_mtime:
            continue

        tasks.append((avro_path, out_pkl))

    print(f"[INFO] 需要生成/更新 1min 缓存：{len(tasks)} 个（缓存目录：{cache_dir}）")

    if tasks:
        t_all = time.perf_counter()
        done_bytes = 0
        done_files = 0

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(avro_file_to_resampled_pkl, a, p, freq): (a, p) for a, p in tasks}

            try:
                for fut in as_completed(futures):
                    avro_path, out_pkl = futures[fut]
                    done_files += 1
                    try:
                        meta = fut.result()
                        done_bytes += avro_path.stat().st_size

                        elapsed = time.perf_counter() - t_all
                        mbps = (done_bytes / 1024 / 1024) / max(elapsed, 1e-9)
                        eta = (elapsed / done_files) * (len(tasks) - done_files)

                        print(
                            f"[CACHE {done_files:>4}/{len(tasks)}] {avro_path.name} -> {out_pkl.name} | "
                            f"t={meta['secs']:.2f}s | 总速={mbps:.2f}MB/s | ETA≈{eta/60:.1f}min"
                        )
                    except Exception as e:
                        print(f"[ERR] 缓存生成失败：{avro_path.name} | {e}")

            except KeyboardInterrupt:
                print("\n[STOP] 收到 Ctrl+C：停止后续缓存任务（已完成的缓存保留）。")

    # 返回存在的 pkl
    pkl_paths = []
    for filename in list_found:
        p = cache_dir / f"{filename}.{freq}.pkl"
        if p.exists():
            pkl_paths.append(p)
    return pkl_paths


# =========================
# 2) 从 1min 缓存 pkl 读取并合并成宽表
# =========================
def merge_from_cached_pkls(pkl_paths: list[Path], cache_freq: str = CACHE_FREQ) -> pd.DataFrame:
    """
    读每个 pkl（index=ts, col0 列）合并成宽表
    列名先用 avro 文件名（从 pkl 文件名解析）
    """
    series_list = []
    suffix = f".{cache_freq}.pkl"

    for pkl in pkl_paths:
        df1 = pd.read_pickle(pkl)  # df1: 单列 col0
        if "col0" not in df1.columns:
            continue

        s = df1["col0"]
        name = pkl.name[:-len(suffix)] if pkl.name.endswith(suffix) else pkl.stem
        s.name = name
        series_list.append(s)

    if not series_list:
        raise ValueError("没有可用的 1min 缓存 pkl 数据。")

    df = pd.concat(series_list, axis=1).sort_index()
    return df


def getdata(max_workers: int = 2):
    # ---------- 读取 mapping ----------
    avro_mapping_path = MAPPING_DIR / f"avro_mapping_{KEY}.csv"
    avro_mapping = pd.read_csv(avro_mapping_path)

    tb_to_file = dict(zip(avro_mapping["datacode/测点编码"], avro_mapping["file_name"]))

    # ---------- 选择最新 need xlsx ----------
    files = list(NEED_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"未在目录 {NEED_DIR} 找到任何 .xlsx 文件")
    selected_xlsx = max(files, key=lambda f: f.stat().st_mtime)

    # outcome 输出：自动加时间戳防覆盖
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    stem = selected_xlsx.stem
    # out_pkl_name = f"{stem}_{FREQUENCY}_{ts_tag}.pkl"
    # out_csv_name = f"{stem}_{FREQUENCY}_{ts_tag}.csv"
    out_pkl_name = f"{stem}_{FREQUENCY}.pkl"
    out_csv_name = f"{stem}_{FREQUENCY}.csv"


    # ---------- 读取 need ----------
    selected_points = pd.read_excel(selected_xlsx)
    selected_points["datacode/测点编码"] = (
        selected_points["datacode/测点编码"].astype(str).str.replace(".", "_", regex=False)
    )

    list_found, list_missing = [], []
    for _, row in selected_points.iterrows():
        code = row.get("datacode/测点编码")
        name = row.get("name/测点名称")
        if code in tb_to_file:
            list_found.append(tb_to_file[code])
        else:
            list_missing.append(name)

    # 去重但保持顺序
    list_found = list(dict.fromkeys(list_found))

    print(f"[INFO] BASE_DIR      : {BASE_DIR}")
    print(f"[INFO] KEY_DIR       : {KEY_DIR}")
    print(f"[INFO] need文件       : {selected_xlsx.name}")
    print(f"[INFO] 需要的 avro 数  : {len(list_found)}")
    if list_missing:
        print("[WARN] 未找到映射（仅显示前20）：", list_missing[:20])

    # ---------- Stage A：长期缓存（永远 1min） ----------
    PKL_DIR.mkdir(parents=True, exist_ok=True)
    pkl_paths = build_persistent_cache_for_list(
        list_found=list_found,
        cache_dir=PKL_DIR,
        freq=CACHE_FREQ,
        max_workers=max_workers,
    )

    # ---------- Stage B：合并为 1min 宽表 ----------
    t0 = time.perf_counter()
    df_1min = merge_from_cached_pkls(pkl_paths, cache_freq=CACHE_FREQ)
    print(f"[INFO] 1min 合并完成：shape={df_1min.shape} 用时 {time.perf_counter()-t0:.2f}s")

    # ---------- Stage C：outcome 按 FREQUENCY 输出 ----------
    if str(FREQUENCY).lower() != str(CACHE_FREQ).lower():
        df = df_1min.resample(FREQUENCY).mean()
    else:
        df = df_1min
    
    df.index.name = "st"

    # # 重命名列：file_name -> datacode（你之前也这样做）
    # rename_dict = dict(zip(avro_mapping["file_name"], avro_mapping["datacode/测点编码"]))
    # df = df.rename(columns=rename_dict)

    # 重命名列：file_name -> datacode（你之前也这样做）
    rename_dict = dict(zip(avro_mapping["file_name"], avro_mapping["name/测点名称"]))
    df = df.rename(columns=rename_dict)
    
    # TIMERANGE 对齐
    if isinstance(TIMERANGE, str) and TIMERANGE == "ALL":
        index = pd.date_range(
            start=df.index.min().floor(FREQUENCY),
            end=df.index.max().ceil(FREQUENCY),
            freq=FREQUENCY,
            tz=df.index.tz,
        )
        df = df.reindex(index)
    else:
        df = df.reindex(TIMERANGE)

    # ---------- 输出 ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path_pkl = OUT_DIR / out_pkl_name
    out_path_csv = OUT_DIR / out_csv_name

    df.to_pickle(out_path_pkl)
    df.to_csv(out_path_csv, encoding="utf-8-sig")

    print("[DONE] 合并结果已保存：", out_path_pkl)
    print("[DONE] CSV 已保存：", out_path_csv)


if __name__ == "__main__":
    getdata(max_workers=2)
