#include "vectordb/FlatIndex.hpp"
#include <algorithm>
#include <queue>

namespace vectordb {

void FlatIndex::AddVector(LabelType label, VectorView vector) {
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
}

std::vector<SearchResult> FlatIndex::Search(VectorView query, size_t k) const {
  if (query.size() != dim_) {
    throw std::invalid_argument("Query dimension mismatch");
  }

  std::shared_lock lock(rw_lock_);
  if (k == 0 || labels_.empty()) return {};

  auto compare = [](const SearchResult &a, const SearchResult &b) {
    return a.distance < b.distance;
  };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(compare)> top_k(compare);

  for (size_t i = 0; i < labels_.size(); ++i) {
    VectorView current_vector(&data_[i * dim_], dim_);
    float dist = dist_calc_(query, current_vector);

    if (top_k.size() < k) {
      top_k.push({labels_[i], dist});
    } else if (dist < top_k.top().distance) {
      top_k.pop();
      top_k.push({labels_[i], dist});
    }
  }

  std::vector<SearchResult> results;
  results.reserve(top_k.size());
  while (!top_k.empty()) {
    results.push_back(top_k.top());
    top_k.pop();
  }
  std::reverse(results.begin(), results.end());
  return results;
}

} // namespace vectordb
