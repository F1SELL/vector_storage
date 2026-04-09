#pragma once

#include "IVectorIndex.hpp"
#include "Distance.hpp"
#include <vector>
#include <shared_mutex>

namespace vectordb {

/**
 * @brief Naive brute-force vector index.
 *
 * Stores vectors in a contiguous memory block for maximum cache efficiency.
 * Search complexity: O(N * D).
 */
class FlatIndex : public IVectorIndex {
 public:
  FlatIndex(size_t dim, MetricType metric)
      : dim_(dim), dist_calc_(dim, metric) {}

  void AddVector(LabelType label, VectorView vector) override;
  [[nodiscard]] std::vector<SearchResult> Search(VectorView query, size_t k) const override;

  [[nodiscard]] size_t Size() const noexcept override {
    std::shared_lock lock(rw_lock_);
    return labels_.size();
  }

  [[nodiscard]] size_t Dimension() const noexcept override { return dim_; }

 private:
  size_t dim_;
  DistanceCalculator dist_calc_;
  std::vector<LabelType> labels_;
  std::vector<float> data_;
  mutable std::shared_mutex rw_lock_;
};

} // namespace vectordb
