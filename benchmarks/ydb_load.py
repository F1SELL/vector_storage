import argparse
import random
import struct
from typing import Iterable

import ydb
import os

def _make_vector(dim: int, rng: random.Random) -> list[float]:
    return [rng.random() for _ in range(dim)]


def _vector_to_binary(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _quote_ydb_path(path: str) -> str:
    escaped = path.replace("`", "``")
    return f"`{escaped}`"


def _batched(iterable: Iterable[dict], batch_size: int) -> Iterable[list[dict]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Load random vectors into YDB.")
    parser.add_argument("--endpoint", default="grpc://localhost:2136")
    parser.add_argument("--database", default="/local")
    parser.add_argument("--table", default="vectors")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--truncate", action="store_true", help="Delete all rows before insert")
    args = parser.parse_args()

    table_path = args.table
    if not table_path.startswith("/"):
        table_path = f"{args.database.rstrip('/')}/{table_path}"
    table_expr = _quote_ydb_path(table_path)

    rng = random.Random(args.seed)
    rows_iter = (
        {
            "id": i,
            "embedding": _vector_to_binary(_make_vector(args.dim, rng)),
        }
        for i in range(args.rows)
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

    if args.truncate:
        pool = ydb.SessionPool(driver)

        def _execute(session: ydb.Session):
            return session.transaction().execute(f"DELETE FROM {table_expr};", commit_tx=True)

        pool.retry_operation_sync(_execute)

    table_client = ydb.TableClient(driver)
    pool = ydb.SessionPool(driver)

    upsert_query = f"""
        DECLARE $batch AS List<Struct<id: Uint64, embedding: String>>;
        UPSERT INTO {table_expr} SELECT id, embedding FROM AS_TABLE($batch);
    """

    total = 0
    for batch in _batched(rows_iter, args.batch):
        def _upsert_callee(current_batch):
            def _execute(session: ydb.Session):
                prepared = session.prepare(upsert_query)
                session.transaction().execute(prepared, {"$batch": current_batch}, commit_tx=True)
            return _execute

        pool.retry_operation_sync(_upsert_callee(batch))
        total += len(batch)
        print(f"Upserted {total}/{args.rows} rows", flush=True)

    driver.stop()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

