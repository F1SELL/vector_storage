import argparse
import json
import os
import statistics
import sys
import time
from typing import Iterable

import ydb


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


def _local_latency_ms(json_path: str, benchmark_substring: str) -> float:
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    benches = payload.get("benchmarks", [])
    has_mean = any(b.get("name", "").endswith("_mean") for b in benches)

    values = []
    for bench in benches:
        name = bench.get("name", "")
        if benchmark_substring not in name:
            continue
        if has_mean and not name.endswith("_mean"):
            continue
        time_unit = bench.get("time_unit", "ns")
        real_time = bench.get("real_time", 0.0)
        values.append(_time_to_ms(real_time, time_unit))

    if not values:
        raise ValueError(f"No benchmarks matching '{benchmark_substring}' found")

    return statistics.median(values)


def _run_ydb_query(pool: ydb.SessionPool, query: str) -> None:
    def _execute(session: ydb.Session):
        return session.transaction().execute(query, commit_tx=True)

    pool.retry_operation_sync(_execute)


def _collect_latencies(pool: ydb.SessionPool, query: str, runs: int) -> Iterable[float]:
    for _ in range(runs):
        start = time.perf_counter()
        _run_ydb_query(pool, query)
        end = time.perf_counter()
        yield (end - start) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local HNSW latency with YDB latency.")
    parser.add_argument("--local-benchmark-json", required=True, help="Path to local benchmark JSON")
    parser.add_argument("--max-ratio", type=float, default=1.15, help="Max allowed local/YDB ratio")
    parser.add_argument("--runs", type=int, default=30, help="Number of YDB query runs")
    parser.add_argument("--benchmark-name", default="HnswIndex_Search", help="Benchmark name substring")
    args = parser.parse_args()

    endpoint = os.environ.get("YDB_ENDPOINT")
    database = os.environ.get("YDB_DATABASE")
    token = os.environ.get("YDB_TOKEN")
    query = os.environ.get("YDB_BENCH_SQL")

    if not endpoint or not database or not query:
        raise ValueError("YDB_ENDPOINT, YDB_DATABASE, and YDB_BENCH_SQL must be set")

    local_ms = _local_latency_ms(args.local_benchmark_json, args.benchmark_name)

    try:
        credentials = ydb.AccessTokenCredentials(token) if token else ydb.credentials_from_env_variables()
    except AttributeError:
        credentials = ydb.AccessTokenCredentials(token) if token else ydb.credentials_from_env()
    driver = ydb.Driver(ydb.DriverConfig(endpoint, database, credentials=credentials))
    driver.wait(fail_fast=True, timeout=10)
    pool = ydb.SessionPool(driver)

    latencies = list(_collect_latencies(pool, query, args.runs))
    ydb_mean = statistics.mean(latencies)
    ydb_median = statistics.median(latencies)

    ratio = local_ms / ydb_mean if ydb_mean > 0 else float("inf")
    print(f"Local HNSW median (ms): {local_ms:.4f}")
    print(f"YDB mean (ms): {ydb_mean:.4f}, median (ms): {ydb_median:.4f}")
    print(f"Local/YDB ratio: {ratio:.3f}, max allowed: {args.max_ratio:.3f}")

    driver.stop()

    if ratio > args.max_ratio:
        print("Comparison failed: local latency exceeds allowed ratio.")
        return 1

    print("Comparison passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
