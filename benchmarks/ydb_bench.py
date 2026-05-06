import argparse
import random
import statistics
import time
import os
from typing import Iterable

import ydb


def _make_vector(dim: int, rng: random.Random) -> list[float]:
    return[rng.random() for _ in range(dim)]


def _quote_ydb_path(path: str) -> str:
    escaped = path.replace("`", "``")
    return f"`{escaped}`"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round((pct / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[idx]


def _collect_latencies(
        pool: ydb.SessionPool,
        query: str,
        params_iter: Iterable[dict],
        warmup: int,
) -> list[float]:
    params_list = list(params_iter)
    latencies_result =[]

    def _benchmark(session: ydb.Session):
        nonlocal latencies_result
        latencies_result.clear()

        prepared = session.prepare(query)

        for i, params in enumerate(params_list):
            start = time.perf_counter()
            session.transaction().execute(prepared, params, commit_tx=True)
            end = time.perf_counter()

            if i >= warmup:
                latencies_result.append((end - start) * 1000.0)

    pool.retry_operation_sync(_benchmark)
    return latencies_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure YDB ANN query latency.")
    parser.add_argument("--endpoint", default="grpc://localhost:2136")
    parser.add_argument("--database", default="/local")
    parser.add_argument("--table", default="vectors")
    parser.add_argument("--index", default="ann_index")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-size", type=int, required=True)
    parser.add_argument("--csv-out", help="Append results to CSV")
    parser.add_argument(
        "--distance-func",
        default="Knn::EuclideanDistance",
    )
    args = parser.parse_args()

    table_path = args.table
    if not table_path.startswith("/"):
        table_path = f"{args.database.rstrip('/')}/{table_path}"
    table_expr = _quote_ydb_path(table_path)

    # ВОТ ОНО - ГЛАВНОЕ ИСПРАВЛЕНИЕ ЗАПРОСА
    query_text = (
        f"DECLARE $q AS List<Float>;\n"
        f"DECLARE $k AS Uint64;\n"
        f"$target = Knn::ToBinaryStringFloat($q);\n"
        f"SELECT id FROM {table_expr} VIEW {args.index} "
        f"ORDER BY {args.distance_func}(embedding, $target) LIMIT $k;"
    )

    rng = random.Random(args.seed)
    query_vectors =[_make_vector(args.dim, rng) for _ in range(args.queries)]

    params_iter = (
        {
            "$q": vec,
            "$k": args.k,
        }
        for vec in query_vectors
    )

    token = os.environ.get("YDB_TOKEN")
    if not token:
        raise SystemExit("Error: YDB_TOKEN environment variable is not set!")

    driver = ydb.Driver(ydb.DriverConfig(
        args.endpoint,
        args.database,
        credentials=ydb.AuthTokenCredentials(token)
    ))
    driver.wait(fail_fast=True, timeout=10)
    pool = ydb.SessionPool(driver)

    print(f"Running benchmark: {args.queries} queries, {args.warmup} warmup...")
    latencies = _collect_latencies(pool, query_text, params_iter, args.warmup)
    driver.stop()

    if not latencies:
        raise SystemExit("No latency samples collected.")

    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    p95_ms = _percentile(latencies, 95.0)

    print(f"YDB ANN latency (ms): mean={mean_ms:.4f}, median={median_ms:.4f}, p95={p95_ms:.4f}")
    print(f"Samples: {len(latencies)} (warmup skipped: {args.warmup})")

    if args.csv_out:
        from pathlib import Path
        import csv

        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()

        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["name", "dataset_size", "latency_ms", "cpu_ms", "iterations", "variant"],
            )
            if write_header:
                writer.writeheader()
            writer.writerow({
                "name": f"IndexFixture/YdbIndex_Search/{args.dataset_size}",
                "dataset_size": args.dataset_size,
                "latency_ms": f"{median_ms:.6f}",
                "cpu_ms": f"{median_ms:.6f}",
                "iterations": len(latencies),
                "variant": "ydb",
            })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())