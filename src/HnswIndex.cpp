#include "vectordb/HnswIndex.hpp"
#include <stdexcept>

namespace vectordb {

HnswIndex::HnswIndex(size_t dim, MetricType metric, Config config)
    : dim_(dim), config_(config) {
  (void)metric;
}

void HnswIndex::AddVector(LabelType label, VectorView vector) {
  if (vector.size() != dim_) {
    throw std::invalid_argument("Vector dimension mismatch");
  }

  std::unique_lock lock(rw_lock_);
  size_t old_data_size = data_.size();
  try {
    data_.insert(data_.end(), vector.begin(), vector.end());
    labels_.push_back(label);
  } catch (...) {
    data_.resize(old_data_size);
    throw;
  }
  count_.fetch_add(1, std::memory_order_relaxed);
}

std::vector<SearchResult> HnswIndex::Search(VectorView query, size_t k) const {
  if (query.size() != dim_) {
    throw std::invalid_argument("Query dimension mismatch");
  }
  if (k == 0) {
    return {};
  }
  throw std::logic_error("HnswIndex search is not implemented yet");
}

} // namespace vectordb
