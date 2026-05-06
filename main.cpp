#include "vectordb/FlatIndex.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace vectordb;

void run_benchmark() {
  const size_t dim = 128;
  const size_t num_vectors = 10000;
  const size_t num_queries = 100;
  const size_t k = 10;


  std::cout << "Dimensions:   " << dim << "\n";
  std::cout << "Dataset size: " << num_vectors << "\n";
  std::cout << "Queries:      " << num_queries << "\n";
  std::cout << "K-NN:         " << k << "\n\n";

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::vector<std::vector<float>> data(num_vectors, std::vector<float>(dim));
  for (auto& vec : data) {
    for (float& val : vec) val = dist(rng);
  }

  std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
  for (auto& vec : queries) {
    for (float& val : vec) val = dist(rng);
  }

  FlatIndex flat_index(dim, MetricType::L2Squared);

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_vectors; ++i) {
    flat_index.AddVector(i, data[i]);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto flat_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::vector<std::vector<SearchResult>> flat_results(num_queries);
  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_queries; ++i) {
    flat_results[i] = flat_index.Search(queries[i], k);
  }
  end = std::chrono::high_resolution_clock::now();
  auto flat_search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

  std::cout << std::left << std::setw(15) << "Index"
            << std::setw(15) << "Build Time"
            << std::setw(25) << "Search Time (100 queries)" << "\n";
  std::cout << std::string(55, '-') << "\n";
  std::cout << std::left << std::setw(15) << "FlatIndex"
            << flat_build_time << " ms\t"
            << flat_search_time << " ms\n\n";

  std::cout << "==========================================\n";
  std::cout << " Sample Query [0] Top-" << std::min<size_t>(5, k) << " Results\n";
  std::cout << "==========================================\n";
  std::cout << std::left << std::setw(10) << "Rank" << std::setw(15) << "Vector ID" << "Distance\n";
  std::cout << std::string(42, '-') << "\n";

  for (size_t i = 0; i < std::min<size_t>(5, k); ++i) {
    std::cout << std::left << std::setw(10) << (i + 1)
              << std::setw(15) << flat_results[0][i].label
              << std::fixed << std::setprecision(4) << flat_results[0][i].distance << "\n";
  }
}

int main() {
  try {
    run_benchmark();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
