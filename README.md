# VectorDB Core

Минимальный in-memory движок векторного поиска на C++23.

## Текущий статус проекта

- `FlatIndex` — полностью рабочий точный индекс (brute-force).
- `HnswIndex` — заготовка публичного API для следующего этапа:
  - `AddVector()` реализован;
  - `Search()` пока не реализован и выбрасывает `std::logic_error`.

## Что сейчас работает

### 1) `FlatIndex`

`FlatIndex` хранит все векторы в непрерывном массиве `float` и выполняет полный перебор:

1. для каждого вектора в базе считает расстояние до запроса;
2. поддерживает top-`k` через `priority_queue`;
3. возвращает результаты, отсортированные по возрастанию дистанции.

Сложность поиска: `O(N * D)`, где:
- `N` — число векторов в индексе,
- `D` — размерность вектора.

### 2) Метрики расстояния

Поддержаны:
- `L2Squared`
- `DotProduct` (как расстояние `-dot`)
- `Cosine`

### 3) SIMD-ускорение

Для `L2Squared` и `DotProduct`:
- AVX-512 на x86 (`__AVX512F__`);
- NEON на ARM (`__ARM_NEON`).

Для ARM добавлен переключатель:
- `-DVECTORDB_DISABLE_NEON=OFF` — NEON включен;
- `-DVECTORDB_DISABLE_NEON=ON` — NEON принудительно выключен (scalar path).

## Тесты

Тесты покрывают:
- точное попадание в себя для `FlatIndex` (дистанция ~ 0);
- корректную сортировку top-`k` по расстоянию;
- проверки на mismatch размерности;
- поведение заготовки `HnswIndex` (бросает `logic_error` в `Search`).

Фактический результат последнего прогона:
- `4/4` теста пройдено;
- `0` падений.

Команды:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Бенчмарк и замеры NEON

Используется `vectordb_bench` (`google/benchmark`), сценарий: `FlatIndex_Search`.

Запуск:

```bash
cmake -S . -B build-neon-on -DCMAKE_BUILD_TYPE=Release -DVECTORDB_DISABLE_NEON=OFF
cmake --build build-neon-on -j
./build-neon-on/vectordb_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true

cmake -S . -B build-neon-off -DCMAKE_BUILD_TYPE=Release -DVECTORDB_DISABLE_NEON=ON
cmake --build build-neon-off -j
./build-neon-off/vectordb_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

### Результаты (CPU mean)

| N (размер базы) | NEON ON | NEON OFF | Ускорение |
|---|---:|---:|---:|
| 100 | 2316 ns | 5604 ns | ~2.42x |
| 1000 | 15992 ns | 51976 ns | ~3.25x |
| 10000 | 143503 ns | 502306 ns | ~3.50x |

Вывод: на ARM включенный NEON дает стабильное ускорение поиска `FlatIndex`, особенно на больших `N`.
