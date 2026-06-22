import json
import math
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ONLINE_ROOT = ROOT / "Online"
sys.path.insert(0, str(ONLINE_ROOT / "data_precessing"))

from online_input import canonical_name, is_missing, read_point_name_rows  # noqa: E402


MAPPING_PATH = ONLINE_ROOT / "input" / "example" / "input_parameter.xlsx"
SOURCE_CSV = ROOT / "datareader_new" / "jqrd" / "outcome" / "上线选取参数_1min.csv"
FALLBACK_SOURCE_PKL = ROOT / "datareader_new" / "jqrd" / "outcome" / "jqrd全场参数_1min.pkl"
OUT_DIR = ONLINE_ROOT / "input" / "test"
RESULT_DIR = ONLINE_ROOT / "output" / "test"


def normalise_source_name(name):
    text = str(name).strip()
    text = re.sub(r"^#\d+锅炉", "", text)
    text = text.replace("TEMP EXHAUSTAVER", "TEMP EXHAUST AVER")
    return canonical_name(text)


def json_value(value):
    if is_missing(value):
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
    except TypeError:
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def build_source_positions(columns, point_rows):
    positions_by_name = {}
    for pos, column in enumerate(columns):
        key = normalise_source_name(column)
        positions_by_name.setdefault(key, []).append(pos)

    used = {}
    resolved = []
    missing_names = []
    for code, display_name in point_rows:
        key = normalise_source_name(display_name)
        positions = positions_by_name.get(key, [])
        occurrence = used.get(key, 0)
        used[key] = occurrence + 1
        if positions:
            resolved.append((code, display_name, positions[min(occurrence, len(positions) - 1)]))
        else:
            resolved.append((code, display_name, None))
            missing_names.append(display_name)
    return resolved, missing_names


def select_rows(df, resolved, count=100):
    positions = [pos for _, _, pos in resolved if pos is not None]
    if not positions:
        return list(df.index[:count])

    completeness = df.iloc[:, positions].notna().sum(axis=1)
    good = completeness[completeness > 0].nlargest(count)
    if good.empty:
        good = completeness.nlargest(count)
    return list(good.index)


def frame_from_row(df, timestamp, resolved):
    row = df.loc[timestamp]
    frame = {"timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")}
    for code, _, pos in resolved:
        frame[code] = None if pos is None else json_value(row.iloc[pos])
    return frame


def dump_json(path, point_table, frames):
    payload = {
        "point_table": point_table,
        "frames": frames,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def dump_preprocess_result(input_path, result_path):
    from online_input import build_online_dataframe_from_file

    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    frames = payload.get("frames", [])
    processed_frames = []
    for frame_index in range(len(frames)):
        df, timestamp, diagnostics = build_online_dataframe_from_file(
            input_path,
            xlsx_path=MAPPING_PATH,
            frame_index=frame_index,
        )
        records = df.where(pd.notna(df), None).to_dict(orient="records")
        processed_frames.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "records": records,
                "diagnostics": diagnostics,
            }
        )
    payload = {
        "source_input": str(input_path.relative_to(ROOT)),
        "frame_count": len(frames),
        "processed_frames": processed_frames,
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def read_source_frame():
    if SOURCE_CSV.exists():
        df = pd.read_csv(SOURCE_CSV, index_col=0, parse_dates=True)
        return SOURCE_CSV, df
    df = pd.read_pickle(FALLBACK_SOURCE_PKL)
    return FALLBACK_SOURCE_PKL, df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    point_rows = read_point_name_rows(MAPPING_PATH)
    point_table = [code for code, _ in point_rows]
    source_path, df = read_source_frame()
    resolved, missing_names = build_source_positions(df.columns, point_rows)
    selected = select_rows(df, resolved, count=100)
    real_frames = [frame_from_row(df, timestamp, resolved) for timestamp in selected]

    stale_patterns = ["real_sample_*.json", "real_samples_from_datareader.json", "real_sample_single_frame.json"]
    for pattern in stale_patterns:
        for path in OUT_DIR.glob(pattern):
            path.unlink()
    stale_result_patterns = [
        "real_sample_*_preprocess_result.json",
        "real_samples_from_datareader_preprocess_result.json",
        "real_sample_single_frame_preprocess_result.json",
    ]
    for pattern in stale_result_patterns:
        for path in RESULT_DIR.glob(pattern):
            path.unlink()

    for sample_index, frame in enumerate(real_frames, start=1):
        name = f"real_sample_{sample_index:03d}"
        frames = [frame]
        input_path = OUT_DIR / f"{name}.json"
        result_path = RESULT_DIR / f"{name}_preprocess_result.json"
        dump_json(input_path, point_table, frames)
        dump_preprocess_result(input_path, result_path)

    summary = {
        "source": str(source_path.relative_to(ROOT)),
        "mapping": str(MAPPING_PATH.relative_to(ROOT)),
        "point_count": len(point_table),
        "real_frame_count": len(real_frames),
        "single_frame_file_count": len(real_frames),
        "selected_timestamps": [frame["timestamp"] for frame in real_frames],
        "unmatched_mapping_names": sorted(set(missing_names)),
    }
    summary["input_dir"] = str(OUT_DIR.relative_to(ROOT))
    summary["result_dir"] = str(RESULT_DIR.relative_to(ROOT))
    (OUT_DIR / "sample_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULT_DIR / "sample_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
