import argparse
import csv
import json
import os
import re
import sys


def _time_to_ms(value: float, unit: str) -> float:
    factors = {
        "ns": 1e-6,
        "us": 1e-3,
        "ms": 1.0,
        "s": 1e3,
    }
    if unit not in factors:
        raise ValueError(f"Unsupported time unit: {unit}")
    return value * factors[unit]


def _strip_aggregate_suffix(name: str) -> str:
    for suffix in ("_mean", "_median", "_stddev", "_cv", "_p50", "_p90", "_p99"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _extract_dataset_size(name: str) -> int | None:
    match = re.search(r"/(\d+)$", name)
    if match:
        return int(match.group(1))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert google/benchmark JSON to CSV.")
    parser.add_argument("input_json", help="Path to benchmark JSON file")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument("--variant", default="default", help="Label for SIMD/scalar variant")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    benches = payload.get("benchmarks", [])
    if not benches:
        raise ValueError("No benchmarks found in JSON")

    has_mean = any(b.get("name", "").endswith("_mean") for b in benches)

    rows = []
    for bench in benches:
        name = bench.get("name", "")
        if has_mean and not name.endswith("_mean"):
            continue

        cleaned_name = _strip_aggregate_suffix(name)
        dataset_size = _extract_dataset_size(cleaned_name)
        time_unit = bench.get("time_unit", "ns")
        real_time = bench.get("real_time", 0.0)
        cpu_time = bench.get("cpu_time", 0.0)

        rows.append({
            "name": cleaned_name,
            "dataset_size": dataset_size if dataset_size is not None else "",
            "latency_ms": _time_to_ms(real_time, time_unit),
            "cpu_ms": _time_to_ms(cpu_time, time_unit),
            "iterations": bench.get("iterations", 0),
            "variant": args.variant,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "dataset_size", "latency_ms", "cpu_ms", "iterations", "variant"])
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
