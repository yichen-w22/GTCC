import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLANT_MODEL_PATH = ROOT / "energy_analysis" / "PLANT_model.py" / "plant_model.py"
DEFAULT_TESTDATA_DIR = ROOT / "testdata"
DEFAULT_OUTPUT_DIR = ROOT / "output"


def load_plant_model():
    spec = importlib.util.spec_from_file_location("plant_model", PLANT_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_build_gases_from_row():
    from energy_analysis.working_fluid.streams import build_gases_from_row

    return build_gases_from_row


def is_missing(value):
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def safe_get(getter):
    try:
        return getter(), None
    except Exception:
        return None, sys.exc_info()[1]


def invalid_state_reasons(streams, gases):
    reasons = Counter()

    for name, state in streams.items():
        missing = []
        for attr in ("P", "T", "m_dot", "h", "s"):
            if is_missing(getattr(state, attr, None)):
                missing.append(attr)
        if missing:
            reasons[f"invalid_stream:{name}:{','.join(missing)}"] += 1

    for name, state in gases.items():
        missing = []
        for attr in ("P", "T", "m_dot", "h", "s"):
            if is_missing(getattr(state, attr, None)):
                missing.append(attr)
        if missing:
            reasons[f"invalid_gas:{name}:{','.join(missing)}"] += 1

    return reasons


def evaluate_sample(pm, input_path):
    sample = {
        "input": str(input_path.relative_to(ROOT)),
        "timestamp": None,
        "status": "ok",
        "error": None,
        "missing_internal_fields": [],
        "gt_running": {},
        "output_total": len(pm.OUTPUT_MAPPING),
        "output_non_null": 0,
        "output_null": len(pm.OUTPUT_MAPPING),
        "null_output_varnames": [],
        "factors": Counter(),
    }

    try:
        df, timestamp, diagnostics = pm.build_online_dataframe_from_file(input_path)
        row = df.iloc[0]
        sample["timestamp"] = timestamp
        sample["missing_internal_fields"] = diagnostics.get("missing_internal_fields", [])
        sample["environment_temperature_cache"] = diagnostics.get("environment_temperature_cache")
        sample["fuel_composition_cache"] = diagnostics.get("fuel_composition_cache")

        for field in sample["missing_internal_fields"]:
            sample["factors"][f"missing_internal_field:{field}"] += 1

        for unit in (1, 2):
            running = pm.is_gt_running(row, unit)
            sample["gt_running"][str(unit)] = running
            if not running:
                sample["factors"][f"gt{unit}_not_running"] += 1

        streams, streams_error = safe_get(lambda: pm.build_streams_from_row(df, 0))
        plant, plant_error = safe_get(lambda: pm.build_plant(df, 0))
        gt1, gt1_error = safe_get(lambda: pm.build_gt(df, 0, 1)) if sample["gt_running"]["1"] else (None, None)
        gt2, gt2_error = safe_get(lambda: pm.build_gt(df, 0, 2)) if sample["gt_running"]["2"] else (None, None)

        if streams is None:
            sample["factors"]["build_streams_failed"] += 1
            if streams_error:
                sample["factors"][f"build_streams_failed:{type(streams_error).__name__}:{streams_error}"] += 1
        if plant is None:
            sample["factors"]["build_plant_failed"] += 1
            if plant_error:
                sample["factors"][f"build_plant_failed:{type(plant_error).__name__}:{plant_error}"] += 1
        if sample["gt_running"]["1"] and gt1 is None:
            sample["factors"]["build_gt1_failed"] += 1
            if gt1_error:
                sample["factors"][f"build_gt1_failed:{type(gt1_error).__name__}:{gt1_error}"] += 1
        if sample["gt_running"]["2"] and gt2 is None:
            sample["factors"]["build_gt2_failed"] += 1
            if gt2_error:
                sample["factors"][f"build_gt2_failed:{type(gt2_error).__name__}:{gt2_error}"] += 1

        if streams is not None:
            sample["factors"].update(invalid_state_reasons(streams, {}))

        gases, gases_error = safe_get(lambda: pm.build_gases_from_row(df, 0))
        if gases is None:
            sample["factors"]["build_gases_failed"] += 1
            if gases_error:
                sample["factors"][f"build_gases_failed:{type(gases_error).__name__}:{gases_error}"] += 1
        else:
            sample["factors"].update(invalid_state_reasons({}, gases))

        metrics = pm.cal_property(df, 0)
        payload = pm.build_output_json(metrics, timestamp=timestamp)
        values = payload["result_point"]
        non_null = [item for item in values if item["value"] is not None]
        nulls = [item for item in values if item["value"] is None]
        sample["output_non_null"] = len(non_null)
        sample["output_null"] = len(nulls)
        sample["null_output_varnames"] = [item["varname"] for item in nulls]

        for item in nulls:
            sample["factors"][f"null_output:{item['varname']}"] += 1

    except Exception as exc:
        sample["status"] = "error"
        sample["error"] = f"{type(exc).__name__}: {exc}"
        sample["factors"][f"exception:{type(exc).__name__}"] += 1

    return sample


def serialize_sample(sample):
    data = dict(sample)
    data["factors"] = dict(sample["factors"])
    return to_jsonable(data)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(to_jsonable(v) for v in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def write_csv(path, samples):
    fieldnames = [
        "input",
        "timestamp",
        "status",
        "error",
        "output_total",
        "output_non_null",
        "output_null",
        "missing_internal_count",
        "missing_internal_fields",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "input": sample["input"],
                    "timestamp": sample["timestamp"],
                    "status": sample["status"],
                    "error": sample["error"],
                    "output_total": sample["output_total"],
                    "output_non_null": sample["output_non_null"],
                    "output_null": sample["output_null"],
                    "missing_internal_count": len(sample["missing_internal_fields"]),
                    "missing_internal_fields": ";".join(sample["missing_internal_fields"]),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate all real_sample_*.json test data.")
    parser.add_argument("--testdata-dir", default=DEFAULT_TESTDATA_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--top", default=10, type=int)
    args = parser.parse_args()

    pm = load_plant_model()
    pm.build_gases_from_row = load_build_gases_from_row()
    input_paths = sorted(args.testdata_dir.glob("real_sample_*.json"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = [evaluate_sample(pm, path) for path in input_paths]
    factor_counts = Counter()
    factor_affected_samples = defaultdict(set)
    for sample in samples:
        factor_counts.update(sample["factors"])
        for factor in sample["factors"]:
            factor_affected_samples[factor].add(sample["input"])

    top_factors = []
    for factor, count in factor_counts.most_common(args.top):
        top_factors.append(
            {
                "factor": factor,
                "count": count,
                "affected_sample_count": len(factor_affected_samples[factor]),
                "example_samples": sorted(factor_affected_samples[factor])[:10],
            }
        )

    report = {
        "sample_count": len(samples),
        "ok_count": sum(1 for sample in samples if sample["status"] == "ok"),
        "error_count": sum(1 for sample in samples if sample["status"] == "error"),
        "average_non_null_outputs": (
            sum(sample["output_non_null"] for sample in samples) / len(samples)
            if samples else 0
        ),
        "top_factors": top_factors,
        "samples": [serialize_sample(sample) for sample in samples],
    }

    json_path = args.output_dir / "testdata_evaluation_report.json"
    csv_path = args.output_dir / "testdata_evaluation_samples.csv"
    json_path.write_text(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, samples)

    print(f"samples={report['sample_count']} ok={report['ok_count']} errors={report['error_count']}")
    print(f"average_non_null_outputs={report['average_non_null_outputs']:.2f}/{len(pm.OUTPUT_MAPPING)}")
    print("top factors:")
    for item in top_factors:
        print(f"- {item['factor']}: count={item['count']}, samples={item['affected_sample_count']}")
    print(f"json_report={json_path}")
    print(f"csv_report={csv_path}")


if __name__ == "__main__":
    main()
