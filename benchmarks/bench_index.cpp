#include <benchmark/benchmark.h>
#include "vectordb/FlatIndex.hpp"
#include "vectordb/HnswIndex.hpp"
#include <random>

using namespace vectordb;

class IndexFixture : public benchmark::Fixture {
 public:
  const size_t dim = 128;
  std::vector<std::vector<float>> data;
  std::vector<float> query;

  void SetUp(const ::benchmark::State& state) override {
    size_t num_vectors = state.range(0);
    data.resize(num_vectors, std::vector<float>(dim));
    query.resize(dim);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& vec : data) {
      for (float& val : vec) val = dist(rng);
    }
    for (float& val : query) val = dist(rng);
  }
};

BENCHMARK_DEFINE_F(IndexFixture, FlatIndex_Search)(benchmark::State& state) {
  FlatIndex index(dim, MetricType::L2Squared);
  for (size_t i = 0; i < data.size(); ++i) {
    index.AddVector(i, VectorView(data[i].data(), dim));
  }

  VectorView q_view(query.data(), dim);
  for (auto _ : state) {
    auto res = index.Search(q_view, 10);
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

BENCHMARK_DEFINE_F(IndexFixture, HnswIndex_Search)(benchmark::State& state) {
  HnswIndex::Config config;
  config.ef_search = 50;
  config.ef_construction = 200;
  HnswIndex index(dim, MetricType::L2Squared, config);
  for (size_t i = 0; i < data.size(); ++i) {
    index.AddVector(i, VectorView(data[i].data(), dim));
  }

  VectorView q_view(query.data(), dim);
  for (auto _ : state) {
    auto res = index.Search(q_view, 10);
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

BENCHMARK_REGISTER_F(IndexFixture, FlatIndex_Search)
    ->RangeMultiplier(10)->Range(100, 10000)->Complexity(benchmark::oN);

BENCHMARK_REGISTER_F(IndexFixture, HnswIndex_Search)
    ->RangeMultiplier(10)->Range(100, 10000)->Complexity(benchmark::oLogN);

BENCHMARK_MAIN();
