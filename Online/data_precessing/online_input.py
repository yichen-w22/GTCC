import json
import math
import os
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLE_DIR = PROJECT_ROOT / "input" / "example"
DEFAULT_MAPPING_PATH = DEFAULT_EXAMPLE_DIR / "input_parameter.xlsx"
DEFAULT_INPUT_PATH = DEFAULT_EXAMPLE_DIR / "input.json"
DEFAULT_CACHE_VALUE_PATH = PROJECT_ROOT / "data_precessing" / "cache_values.json"

P0 = 101325.0
R_MIX = 291.972
DEFAULT_FUEL_DENSITY = 0.764
FUEL_COMPOSITION_COLUMNS = [
    "H2", "N2", "CO2", "CH4", "CO", "O2+Ar",
    "C2H6", "C3H8", "iC4H10", "nC4H10", "iC5H12", "nC5H12",
]
DEFAULT_RUNTIME_CACHE_COLUMNS = [
    "高压缸排汽温度",
    "热网换热量",
    "环境温度_1",
    "环境温度_2",
    "大气相对湿度_1",
    "大气相对湿度_2",
    "压气机入口压力_1",
    "压气机入口压力_2",
    "燃料温度_1",
    "燃料温度_2",
    "冷凝水温度_1",
    "冷凝水温度_2",
    "热网抽汽流量",
    "低压缸进汽温度",
]
DEFAULT_FUEL_COMPOSITION = {
    "H2": 0.033,
    "N2": 2.152,
    "CO2": 0.3305,
    "CH4": 92.85,
    "CO": 0.0086,
    "O2+Ar": 0.065,
    "C2H6": 3.642,
    "C3H8": 0.7565,
    "iC4H10": 0.0212,
    "nC4H10": 0.0828,
    "iC5H12": 0.0249,
    "nC5H12": 0.0181,
}
DEFAULT_CACHE_COLUMNS = DEFAULT_RUNTIME_CACHE_COLUMNS + FUEL_COMPOSITION_COLUMNS
DEFAULT_CACHE_INITIAL_VALUES = {
    **DEFAULT_FUEL_COMPOSITION,
    "环境温度_1": 15.0,
    "环境温度_2": 15.0,
}


BASE_AVERAGING_GROUPS = {
    "压气机入口压力_2": ["PRES U/STR COMPR"],
    "压气机出口温度_2": ["TEMP COMPR OUTLET"],
    "压气机出口压力_2": ["PRES COMPR OUTLET"],
    "燃机出力_2": ["#2燃机发电机功率"],
    "燃料流量_2": ["燃机前置模块天然气进气流量"],
    "燃料温度_2": ["TEMP U/STR NG ESV"],
    "燃料压力_2": ["NG PRES U/STR ESV"],
    "大气相对湿度_2": ["REL HUMDITY AMB AIR"],
    "环境温度_2": ["TEMP AMBIENT AIR"],
    "高压省煤器进水压力_2": ["高压省煤器进水压力"],
    "高压省煤器进水温度_2": ["高压省煤器进水温度"],
    "高压给水流量_2": ["高压给水流量1", "高压给水流量2", "高压给水流量3"],
    "高压主蒸汽温度_2": ["高压主蒸汽温度1", "高压主蒸汽温度2", "高压主蒸汽温度3"],
    "高压主蒸汽压力_2": ["高压主蒸汽压力1", "高压主蒸汽压力2", "高压主蒸汽压力3"],
    "高压主蒸汽流量_2": ["高压主蒸汽流量1", "高压主蒸汽流量2", "高压主蒸汽流量3"],
    "中压省煤器进水压力_2": ["中压省煤器进水压力"],
    "中压给水流量_2": ["中压给水流量1", "中压给水流量2", "中压给水流量3"],
    "热再热蒸汽流量_2": ["热再热蒸汽流量1", "热再热蒸汽流量2", "热再热蒸汽流量3"],
    "热再热蒸汽温度_2": ["热再热蒸汽温度1", "热再热蒸汽温度2", "热再热蒸汽温度3"],
    "热再热蒸汽压力_2": ["热再热蒸汽压力1", "热再热蒸汽压力2", "热再热蒸汽压力3"],
    "低压给水流量_2": ["低压给水流量1", "低压给水流量2", "低压给水流量3"],
    "低压省煤器出水温度_2": ["低压省煤器出水温度"],
    "低压汽包压力_2": ["低压汽包压力1", "低压汽包压力2", "低压汽包压力3"],
    "低压主蒸汽压力_2": ["低压主蒸汽压力1", "低压主蒸汽压力2", "低压主蒸汽压力3"],
    "低压主蒸汽温度_2": ["低压主蒸汽温度1", "低压主蒸汽温度2", "低压主蒸汽温度3"],
    "低压主蒸汽流量_2": ["低压主蒸汽流量1", "低压主蒸汽流量2", "低压主蒸汽流量3"],
    "冷凝水温度_2": ["冷凝水温度"],
    "进口烟温_2": ["左侧进口烟温", "右侧进口烟温"],
    "进口烟压_2": ["进口烟压"],
    "排烟烟压_2": ["排烟烟压"],
    "排烟烟温_2": ["排烟烟温"],
    "余热锅炉出口烟气流量_2": ["余热锅炉出口烟气流量"],
    "压气机入口压力_1": ["PRES U/STR COMPR.1"],
    "压气机出口温度_1": ["TEMP COMPR OUTLET.1", "TEMP COMPR OUTLET.2", "TEMP COMPR OUTLET.3"],
    "压气机出口压力_1": ["PRES COMPR OUTLET.1"],
    "燃机出力_1": ["#1燃机发电机功率"],
    "燃料流量_1": ["1#燃气流量"],
    "燃料温度_1": ["TEMP U/STR NG ESV.1"],
    "燃料压力_1": ["NG PRES U/STR ESV.1"],
    "大气相对湿度_1": ["REL HUMDITY AMB AIR.1"],
    "环境温度_1": ["TEMP AMBIENT AIR.1"],
    "高压省煤器进水压力_1": ["高压省煤器进水压力.1"],
    "高压省煤器进水温度_1": ["高压省煤器进水温度.1"],
    "高压给水流量_1": ["高压给水流量1.1", "高压给水流量2.1", "高压给水流量3.1"],
    "高压主蒸汽温度_1": ["高压主蒸汽温度1.1", "高压主蒸汽温度2.1", "高压主蒸汽温度3.1"],
    "高压主蒸汽压力_1": ["高压主蒸汽压力1.1", "高压主蒸汽压力2.1", "高压主蒸汽压力3.1"],
    "高压主蒸汽流量_1": ["高压主蒸汽流量1.1", "高压主蒸汽流量2.1", "高压主蒸汽流量3.1"],
    "中压省煤器进水压力_1": ["中压省煤器进水压力.1"],
    "中压给水流量_1": ["中压给水流量1.1", "中压给水流量2.1", "中压给水流量3.1"],
    "热再热蒸汽流量_1": ["热再热蒸汽流量1.1", "热再热蒸汽流量2.1", "热再热蒸汽流量3.1"],
    "热再热蒸汽温度_1": ["热再热蒸汽温度1.1", "热再热蒸汽温度2.1", "热再热蒸汽温度3.1"],
    "热再热蒸汽压力_1": ["热再热蒸汽压力1.1", "热再热蒸汽压力2.1", "热再热蒸汽压力3.1"],
    "低压给水流量_1": ["低压给水流量1.1", "低压给水流量2.1", "低压给水流量3.1"],
    "低压省煤器出水温度_1": ["低压省煤器出水温度.1"],
    "低压汽包压力_1": ["低压汽包压力1.1", "低压汽包压力2.1", "低压汽包压力3.1"],
    "低压主蒸汽压力_1": ["低压主蒸汽压力1.1", "低压主蒸汽压力2.1", "低压主蒸汽压力3.1"],
    "低压主蒸汽温度_1": ["低压主蒸汽温度1.1", "低压主蒸汽温度2.1", "低压主蒸汽温度3.1"],
    "低压主蒸汽流量_1": ["低压主蒸汽流量1.1", "低压主蒸汽流量2.1", "低压主蒸汽流量3.1"],
    "冷凝水温度_1": ["冷凝水温度.1"],
    "进口烟温_1": ["左侧进口烟温.1", "右侧进口烟温.1"],
    "进口烟压_1": ["进口烟压.1"],
    "排烟烟压_1": ["排烟烟压.1"],
    "排烟烟温_1": ["排烟烟温.1"],
    "余热锅炉出口烟气流量_1": ["余热锅炉出口烟气流量.1"],
    "高压缸排汽温度": ["HP EXHAUST STEAM TEMP"],
    "高压缸排汽压力": ["高压缸排汽压力"],
    "中压缸排汽温度": ["IP EXHAUST STEAM TEMP"],
    "中压缸排汽压力": ["IP EXHAUST PRESSURE"],
    "低压缸进汽温度": ["LP INLET STEAM TEMP"],
    "低压缸进汽压力": ["LP INLET PRESSURE"],
    "凝汽器压力": ["CONDENSER VACUUM 1", "CONDENSER VACUUM 2", "CONDENSER VACUUM 3"],
    "汽机出力": ["汽机发电机功率"],
    "热网抽汽流量": ["采暖抽汽流量"],
    "热网换热量": ["二期热网瞬时热量"],
    "凝结水泵出口母管压力": ["凝结水泵出口母管压力"],
    "H2": ["H2"],
    "N2": ["N2"],
    "CO2": ["CO2"],
    "CH4": ["CH4"],
    "CO": ["CO"],
    "O2+Ar": ["O2+Ar"],
    "C2H6": ["C2H6"],
    "C3H8": ["C3H8"],
    "iC4H10": ["iC4H10"],
    "nC4H10": ["nC4H10"],
    "iC5H12": ["iC5H12"],
    "nC5H12": ["nC5H12"],
}

PRESSURE_MPA_COLUMNS = [
    "高压省煤器进水压力_2", "高压主蒸汽压力_2", "中压省煤器进水压力_2",
    "热再热蒸汽压力_2", "低压汽包压力_2", "低压主蒸汽压力_2",
    "高压省煤器进水压力_1", "高压主蒸汽压力_1", "中压省煤器进水压力_1",
    "热再热蒸汽压力_1", "低压汽包压力_1", "低压主蒸汽压力_1",
    "高压缸排汽压力", "中压缸排汽压力", "低压缸进汽压力",
    "压气机出口压力_2", "压气机出口压力_1", "燃料压力_2", "燃料压力_1",
    "凝结水泵出口母管压力",
]

TEMPERATURE_COLUMNS = [
    "高压省煤器进水温度_2", "高压主蒸汽温度_2",
    "热再热蒸汽温度_2", "低压省煤器出水温度_2",
    "低压主蒸汽温度_2", "冷凝水温度_2", "进口烟温_2", "排烟烟温_2",
    "高压省煤器进水温度_1", "高压主蒸汽温度_1",
    "热再热蒸汽温度_1", "低压省煤器出水温度_1",
    "低压主蒸汽温度_1", "冷凝水温度_1", "进口烟温_1", "排烟烟温_1",
    "高压缸排汽温度", "中压缸排汽温度", "低压缸进汽温度",
    "环境温度_1", "环境温度_2", "压气机出口温度_2", "燃料温度_2",
    "压气机出口温度_1", "燃料温度_1",
]

FLOW_TPH_COLUMNS = [
    "高压给水流量_2", "高压主蒸汽流量_2", "中压给水流量_2",
    "热再热蒸汽流量_2", "低压给水流量_2", "低压主蒸汽流量_2",
    "高压给水流量_1", "高压主蒸汽流量_1", "中压给水流量_1",
    "热再热蒸汽流量_1", "低压给水流量_1", "低压主蒸汽流量_1",
    "热网抽汽流量",
]


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def to_float(value):
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_binary(a, b, op):
    a = to_float(a)
    b = to_float(b)
    if a is None or b is None:
        return None
    try:
        return op(a, b)
    except Exception:
        return None


def safe_mul(value, factor):
    value = to_float(value)
    return None if value is None else value * factor


def safe_add(value, offset):
    value = to_float(value)
    return None if value is None else value + offset


def canonical_name(name):
    name = str(name).strip()
    name = re.sub(r"\s+\.(\d+)$", r".\1", name)
    name = re.sub(r"\s+", " ", name)
    return name


def read_point_name_rows(xlsx_path=DEFAULT_MAPPING_PATH):
    rows = []
    sheets = pd.read_excel(xlsx_path, sheet_name=None, header=None, dtype=str)
    for sheet in sheets.values():
        if sheet.empty:
            continue
        header = [canonical_name(value) for value in sheet.iloc[0].tolist()]
        if "datacode/测点编码" in header and "name/测点名称" in header:
            code_col = header.index("datacode/测点编码")
            name_col = header.index("name/测点名称")
            data = sheet.iloc[1:, [code_col, name_col]]
        else:
            data = sheet.iloc[:, [0, 1]]
        for code, name in data.itertuples(index=False, name=None):
            code = "" if is_missing(code) else str(code).strip()
            name = "" if is_missing(name) else str(name).strip()
            if code and name and code not in {"JQRD_编码", "datacode/测点编码"}:
                rows.append((code, canonical_name(name)))
    return rows


def parse_input_payload(payload, frame_index=0):
    if isinstance(payload, (str, Path)):
        payload = json.loads(Path(payload).read_text(encoding="utf-8-sig"))
    frames = payload.get("frames") or []
    frame = frames[frame_index] if frame_index < len(frames) else {}
    timestamp = frame.get("timestamp")
    point_values = {key: to_float(value) for key, value in frame.items() if key != "timestamp"}
    return timestamp, point_values


def build_raw_named_values(point_values, point_rows):
    counts = {}
    raw = {}
    code_to_name = {}
    missing_point_codes = []
    for code, base_name in point_rows:
        index = counts.get(base_name, 0)
        counts[base_name] = index + 1
        name = base_name if index == 0 else f"{base_name}.{index}"
        code_to_name[code] = name
        if code not in point_values:
            missing_point_codes.append(code)
        raw[name] = point_values.get(code)
    return raw, code_to_name, missing_point_codes


def value_from_names(raw, names):
    values = []
    for name in names:
        value = raw.get(canonical_name(name))
        if not is_missing(value):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def apply_averaging(raw):
    groups = {key: list(value) for key, value in BASE_AVERAGING_GROUPS.items()}
    row = {new_name: value_from_names(raw, names) for new_name, names in groups.items()}
    missing_groups = {new_name: names for new_name, names in groups.items() if row[new_name] is None}
    return row, missing_groups


def _default_cache_rows():
    rows = []
    for name in DEFAULT_CACHE_COLUMNS:
        initial_value = DEFAULT_CACHE_INITIAL_VALUES.get(name)
        rows.append(
            {
                "name": name,
                "initial_value": initial_value,
                "cached_value": initial_value,
            }
        )
    return rows


def _normalize_cache_rows(payload):
    rows = []
    if isinstance(payload, list):
        source_rows = payload
    elif isinstance(payload, dict):
        source_rows = payload.get("rows", payload.get("values", []))
    else:
        source_rows = []

    for item in source_rows:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        initial_value = to_float(item.get("initial_value"))
        cached_value = to_float(item.get("cached_value"))
        rows.append(
            {
                "name": name,
                "initial_value": initial_value,
                "cached_value": cached_value,
            }
        )
    return rows


def read_cache_rows(cache_path=DEFAULT_CACHE_VALUE_PATH):
    cache_path = Path(cache_path)
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _default_cache_rows(), "default"

    rows = _normalize_cache_rows(payload)
    if not rows:
        return _default_cache_rows(), "default"

    by_name = {row["name"]: row for row in rows}
    merged = []
    for default_row in _default_cache_rows():
        current = by_name.get(default_row["name"], {})
        initial_value = current.get("initial_value", default_row["initial_value"])
        cached_value = current.get("cached_value", default_row["cached_value"])
        merged.append(
            {
                "name": default_row["name"],
                "initial_value": initial_value,
                "cached_value": cached_value,
            }
        )
    return merged, "file"


def write_cache_rows(rows, cache_path=DEFAULT_CACHE_VALUE_PATH):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, cache_path)


def _valid_cache_input(name, value):
    if value is None:
        return False
    if name in FUEL_COMPOSITION_COLUMNS and value < 0:
        return False
    return True


def apply_cache_values(row, cache_path=DEFAULT_CACHE_VALUE_PATH):
    rows, source = read_cache_rows(cache_path)
    updated_fields = []
    filled_fields = []
    initialized_fields = []

    for item in rows:
        name = item["name"]
        value = to_float(row.get(name))
        if _valid_cache_input(name, value):
            item["cached_value"] = value
            updated_fields.append(name)
            continue

        cached_value = to_float(item.get("cached_value"))
        if cached_value is not None:
            row[name] = cached_value
            filled_fields.append(name)
            continue

        initial_value = to_float(item.get("initial_value"))
        if initial_value is not None:
            item["cached_value"] = initial_value
            row[name] = initial_value
            initialized_fields.append(name)

    if updated_fields or initialized_fields or source == "default":
        write_cache_rows(rows, cache_path=cache_path)
        if updated_fields:
            source = "updated"
        elif initialized_fields:
            source = "initialized"
        else:
            source = "default_initialized"

    return {
        "source": source,
        "columns": [item["name"] for item in rows],
        "updated_fields": updated_fields,
        "filled_fields": filled_fields,
        "initialized_fields": initialized_fields,
        "cache_path": str(Path(cache_path)),
    }


def apply_unit_conversions(row):
    for col in PRESSURE_MPA_COLUMNS:
        row[col] = safe_mul(row.get(col), 1_000_000.0)
    row["压气机入口压力_2"] = safe_mul(row.get("压气机入口压力_2"), 100.0)
    row["压气机入口压力_1"] = safe_mul(row.get("压气机入口压力_1"), 100.0)
    for col in ["进口烟压_2", "排烟烟压_2", "进口烟压_1", "排烟烟压_1"]:
        value = safe_mul(row.get(col), 1000.0)
        row[col] = None if value is None else value + P0
    row["凝汽器压力"] = safe_mul(row.get("凝汽器压力"), 1000.0)

    for col in TEMPERATURE_COLUMNS:
        row[col] = safe_add(row.get(col), 273.15)

    for col in FLOW_TPH_COLUMNS:
        row[col] = safe_mul(row.get(col), 1000.0 / 3600.0)

    row["燃机出力_2"] = safe_mul(row.get("燃机出力_2"), 1_000_000.0)
    row["燃机出力_1"] = safe_mul(row.get("燃机出力_1"), 1_000_000.0)
    row["汽机出力"] = safe_mul(row.get("汽机出力"), 1_000_000.0)
    row["热网换热量"] = safe_mul(row.get("热网换热量"), 1.0e9 / 3600.0)
    return row


def add_derived_values(row):
    fuel_density = to_float(row.get("密度")) or DEFAULT_FUEL_DENSITY
    row["燃料质量流量_2"] = safe_mul(row.get("燃料流量_2"), fuel_density / 3600.0)
    row["燃料质量流量_1"] = safe_mul(row.get("燃料流量_1"), fuel_density / 3600.0)

    def flue_mass(unit):
        temp = to_float(row.get(f"排烟烟温_{unit}"))
        volume_flow = to_float(row.get(f"余热锅炉出口烟气流量_{unit}"))
        if temp is None or volume_flow is None:
            return None
        return P0 / (R_MIX * temp) * volume_flow / 3600.0

    row["高压缸排汽进入锅炉_2"] = safe_binary(row.get("热再热蒸汽流量_2"), row.get("中压给水流量_2"), lambda a, b: a - b)
    row["燃机进口空气流量_2"] = safe_binary(flue_mass(2), row.get("燃料质量流量_2"), lambda a, b: a - b)
    row["高压缸排汽进入锅炉_1"] = safe_binary(row.get("热再热蒸汽流量_1"), row.get("中压给水流量_1"), lambda a, b: a - b)
    row["燃机进口空气流量_1"] = safe_binary(flue_mass(1), row.get("燃料质量流量_1"), lambda a, b: a - b)
    row.pop("燃料流量_2", None)
    row.pop("燃料流量_1", None)
    row.pop("余热锅炉出口烟气流量_2", None)
    row.pop("余热锅炉出口烟气流量_1", None)
    return row


def build_online_dataframe(payload, xlsx_path=DEFAULT_MAPPING_PATH, frame_index=0):
    timestamp, point_values = parse_input_payload(payload, frame_index=frame_index)
    point_rows = read_point_name_rows(xlsx_path)
    raw, code_to_name, missing_point_codes = build_raw_named_values(point_values, point_rows)
    row, missing_groups = apply_averaging(raw)
    cache_values = apply_cache_values(row)
    missing_groups = {name: names for name, names in missing_groups.items() if is_missing(row.get(name))}
    apply_unit_conversions(row)
    add_derived_values(row)
    df = pd.DataFrame([row])
    diagnostics = {
        "timestamp": timestamp,
        "point_count": len(point_rows),
        "missing_point_codes": missing_point_codes,
        "missing_internal_fields": sorted(name for name, value in row.items() if value is None),
        "missing_averaging_groups": missing_groups,
        "code_to_name": code_to_name,
        "cache_values": cache_values,
    }
    return df, timestamp, diagnostics


def build_online_dataframe_from_file(input_path=DEFAULT_INPUT_PATH, xlsx_path=DEFAULT_MAPPING_PATH, frame_index=0):
    return build_online_dataframe(Path(input_path), xlsx_path=xlsx_path, frame_index=frame_index)


if __name__ == "__main__":
    df, timestamp, diagnostics = build_online_dataframe_from_file()
    print(f"timestamp={timestamp}")
    print(f"columns={len(df.columns)}")
    print(f"missing_internal_fields={len(diagnostics['missing_internal_fields'])}")
    print(df.head(1).to_string(index=False))
