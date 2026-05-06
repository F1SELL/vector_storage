#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/output_graphs"

BUILD_SIMD="${ROOT_DIR}/build-simd"
BUILD_SCALAR="${ROOT_DIR}/build-scalar"

SIMD_JSON="${ROOT_DIR}/bench_simd.json"
SIMD_CSV="${ROOT_DIR}/bench_simd.csv"
SCALAR_JSON="${ROOT_DIR}/bench_scalar.json"
SCALAR_CSV="${ROOT_DIR}/bench_scalar.csv"
YDB_CSV="${ROOT_DIR}/bench_ydb.csv"

ENDPOINT="grpcs://ydb.serverless.yandexcloud.net:2135"
DATABASE="/ru-central1/b1g7j604594vcgh5rraq/etnjg8ncv94luquv54cd"
TABLE="vectors"
INDEX="ann_index"
DIM="128"
K="10"
QUERIES="100"
WARMUP="10"
DATASET_SIZES="10000"
INSTALL_DEPS="0"
LOAD_YDB="0"
YDB_ROWS="10000"
YDB_BATCH="1000"
YDB_TRUNCATE="0"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --endpoint       YDB endpoint (default: ${ENDPOINT})
  --database       YDB database (default: ${DATABASE})
  --table          YDB table (default: ${TABLE})
  --index          YDB ANN index (default: ${INDEX})
  --dim            Vector dimension (default: ${DIM})
  --k              Top-k (default: ${K})
  --queries        Number of queries (default: ${QUERIES})
  --warmup         Warmup queries to skip (default: ${WARMUP})
  --dataset-sizes  Comma-separated dataset sizes (default: ${DATASET_SIZES})
  --install-deps   Install Python deps from benchmarks/requirements.txt
  --load-ydb       Populate YDB table before running benchmarks
  --ydb-rows       Number of rows to load when --load-ydb is set (default: ${YDB_ROWS})
  --ydb-batch      Batch size for YDB bulk upsert (default: ${YDB_BATCH})
  --ydb-truncate   Delete all rows before loading (default: ${YDB_TRUNCATE})
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2;;
    --database) DATABASE="$2"; shift 2;;
    --table) TABLE="$2"; shift 2;;
    --index) INDEX="$2"; shift 2;;
    --dim) DIM="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --queries) QUERIES="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --dataset-sizes) DATASET_SIZES="$2"; shift 2;;
    --install-deps) INSTALL_DEPS="1"; shift 1;;
    --load-ydb) LOAD_YDB="1"; shift 1;;
    --ydb-rows) YDB_ROWS="$2"; shift 2;;
    --ydb-batch) YDB_BATCH="$2"; shift 2;;
    --ydb-truncate) YDB_TRUNCATE="1"; shift 1;;
    -h|--help) print_usage; exit 0;;
    *) echo "Unknown option: $1"; print_usage; exit 1;;
  esac
done

if [[ "$INSTALL_DEPS" == "1" ]]; then
  python3 -m pip install -r "${ROOT_DIR}/benchmarks/requirements.txt"
fi

if [[ "$LOAD_YDB" == "1" ]]; then
  LOAD_ARGS=(
    --endpoint "${ENDPOINT}"
    --database "${DATABASE}"
    --table "${TABLE}"
    --rows "${YDB_ROWS}"
    --dim "${DIM}"
    --batch "${YDB_BATCH}"
  )
  if [[ "$YDB_TRUNCATE" == "1" ]]; then
    LOAD_ARGS+=(--truncate)
  fi
  python3 "${ROOT_DIR}/benchmarks/ydb_load.py" "${LOAD_ARGS[@]}"
fi

mkdir -p "${OUTPUT_DIR}/simd_vs_scalar" "${OUTPUT_DIR}/flat_vs_hnsw_simd" "${OUTPUT_DIR}/hnsw_vs_ydb"

cmake -S "${ROOT_DIR}" -B "${BUILD_SIMD}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_SIMD}" -j
"${BUILD_SIMD}/vectordb_bench" --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true \
  --benchmark_format=json \
  --benchmark_out="${SIMD_JSON}"
python3 "${ROOT_DIR}/benchmarks/benchmark_to_csv.py" "${SIMD_JSON}" "${SIMD_CSV}" --variant simd

cmake -S "${ROOT_DIR}" -B "${BUILD_SCALAR}" -DCMAKE_BUILD_TYPE=Release -DVECTORDB_DISABLE_SIMD=ON
cmake --build "${BUILD_SCALAR}" -j
"${BUILD_SCALAR}/vectordb_bench" --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true \
  --benchmark_format=json \
  --benchmark_out="${SCALAR_JSON}"
python3 "${ROOT_DIR}/benchmarks/benchmark_to_csv.py" "${SCALAR_JSON}" "${SCALAR_CSV}" --variant scalar

python3 "${ROOT_DIR}/benchmarks/plot_benchmarks.py" --output-dir "${OUTPUT_DIR}/simd_vs_scalar" \
  --inputs "${SIMD_CSV}" "${SCALAR_CSV}" --labels simd scalar

python3 "${ROOT_DIR}/benchmarks/plot_benchmarks.py" --output-dir "${OUTPUT_DIR}/flat_vs_hnsw_simd" "${SIMD_CSV}"

rm -f "${YDB_CSV}"
IFS="," read -r -a SIZE_LIST <<< "${DATASET_SIZES}"
for size in "${SIZE_LIST[@]}"; do
  python3 "${ROOT_DIR}/benchmarks/ydb_bench.py" \
    --endpoint "${ENDPOINT}" \
    --database "${DATABASE}" \
    --table "${TABLE}" \
    --index "${INDEX}" \
    --dim "${DIM}" \
    --k "${K}" \
    --queries "${QUERIES}" \
    --warmup "${WARMUP}" \
    --dataset-size "${size}" \
    --distance-func Knn::EuclideanDistance \
    --csv-out "${YDB_CSV}"
done

python3 "${ROOT_DIR}/benchmarks/plot_benchmarks.py" --output-dir "${OUTPUT_DIR}/hnsw_vs_ydb" \
  --inputs "${SIMD_CSV}" "${YDB_CSV}" --labels local ydb

echo "Graphs saved under: ${OUTPUT_DIR}"
