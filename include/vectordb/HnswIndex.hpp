#pragma once

#include "IVectorIndex.hpp"
#include "Distance.hpp"
#include <shared_mutex>
#include <atomic>
#include <vector>

namespace vectordb {

/**
 * @brief Hierarchical Navigable Small World (HNSW) index.
 */
class HnswIndex : public IVectorIndex {
 public:
  struct Config {
    size_t M = 16;
    size_t M_max0 = 32;
    size_t ef_construction = 200;
    size_t ef_search = 50;
    float level_mult = 1.0f / std::log(1.0f * static_cast<float>(M));

    Config()
        : M(16),
          M_max0(32),
          ef_construction(200),
          ef_search(50),
          level_mult(1.0f / std::log(16.0f)) {}
  };

  HnswIndex(size_t dim, MetricType metric, Config config = Config());

  void AddVector(LabelType label, VectorView vector) override;
  [[nodiscard]] std::vector<SearchResult> Search(VectorView query, size_t k) const override;

  [[nodiscard]] size_t Size() const noexcept override { return count_.load(std::memory_order_relaxed); }
  [[nodiscard]] size_t Dimension() const noexcept override { return dim_; }

 private:
  size_t dim_;
  Config config_;
  std::atomic<size_t> count_{0};
  std::vector<LabelType> labels_;
  std::vector<float> data_;
  mutable std::shared_mutex rw_lock_;
};

} // namespace vectordb
