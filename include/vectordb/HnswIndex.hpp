#pragma once

#include "IVectorIndex.hpp"
#include "Distance.hpp"
#include <atomic>
#include <mutex>
#include <cmath>
#include <random>
#include <shared_mutex>
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

  // Amortized complexity: O(M * ef_construction * log N) for insertion.
  void AddVector(LabelType label, VectorView vector) override;
  // Approximate complexity: O(ef_search * log N) for search.
  [[nodiscard]] std::vector<SearchResult> Search(VectorView query, size_t k) const override;

  [[nodiscard]] size_t Size() const noexcept override { return count_.load(std::memory_order_relaxed); }
  [[nodiscard]] size_t Dimension() const noexcept override { return dim_; }

 private:
  struct Node {
    size_t level = 0;
    std::vector<std::vector<size_t>> links;
  };

  struct Candidate {
    size_t id;
    float distance;
  };

  size_t RandomLevel();
  size_t GreedySearchLevel(VectorView query, size_t entry_id, size_t level) const;
  std::vector<Candidate> SearchLayer(VectorView query, size_t entry_id, size_t ef, size_t level) const;
  std::vector<size_t> SelectNeighborsHeuristic(const std::vector<Candidate>& candidates, size_t max_neighbors) const;
  std::vector<size_t> SelectNeighborsHeuristicFromNode(size_t base_id,
                                                       const std::vector<size_t>& candidates,
                                                       size_t max_neighbors) const;
  void LinkBidirectional(size_t source_id, size_t target_id, size_t level, size_t max_neighbors);
  VectorView VectorById(size_t id) const;

  size_t dim_;
  Config config_;
  DistanceCalculator dist_calc_;

  std::atomic<size_t> count_{0};
  std::vector<LabelType> labels_;
  std::vector<float> data_;
  std::vector<Node> nodes_;

  size_t entry_point_ = 0;
  size_t max_level_ = 0;

  mutable std::shared_mutex rw_lock_;
  std::mt19937 rng_;
  std::uniform_real_distribution<float> level_dist_{0.0f, 1.0f};
};

} // namespace vectordb
