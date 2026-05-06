#include "vectordb/HnswIndex.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>

namespace vectordb {

HnswIndex::HnswIndex(size_t dim, MetricType metric, Config config)
    : dim_(dim),
      config_(config),
      dist_calc_(dim, metric),
      rng_(std::random_device{}()) {
  if (config_.M == 0) {
    throw std::invalid_argument("HNSW M must be > 0");
  }
  if (config_.M_max0 < config_.M) {
    config_.M_max0 = config_.M * 2;
  }
  if (config_.level_mult <= 0.0f) {
    config_.level_mult = 1.0f / std::log(static_cast<float>(config_.M));
  }
}

size_t HnswIndex::RandomLevel() {
  float r = std::max(level_dist_(rng_), std::numeric_limits<float>::min());
  return static_cast<size_t>(-std::log(r) * config_.level_mult);
}

VectorView HnswIndex::VectorById(size_t id) const {
  return VectorView(&data_[id * dim_], dim_);
}

size_t HnswIndex::GreedySearchLevel(VectorView query, size_t entry_id, size_t level) const {
  size_t current = entry_id;
  float current_dist = dist_calc_(query, VectorById(current));
  bool changed = true;

  while (changed) {
    changed = false;
    const auto& neighbors = nodes_[current].links[level];
    for (size_t neighbor : neighbors) {
      float dist = dist_calc_(query, VectorById(neighbor));
      if (dist < current_dist) {
        current_dist = dist;
        current = neighbor;
        changed = true;
      }
    }
  }

  return current;
}

std::vector<HnswIndex::Candidate> HnswIndex::SearchLayer(VectorView query,
                                                         size_t entry_id,
                                                         size_t ef,
                                                         size_t level) const {
  struct MinByDistance {
    bool operator()(const Candidate& a, const Candidate& b) const { return a.distance > b.distance; }
  };
  struct MaxByDistance {
    bool operator()(const Candidate& a, const Candidate& b) const { return a.distance < b.distance; }
  };

  std::priority_queue<Candidate, std::vector<Candidate>, MinByDistance> candidates;
  std::priority_queue<Candidate, std::vector<Candidate>, MaxByDistance> results;
  std::vector<uint8_t> visited(nodes_.size(), 0);

  float entry_dist = dist_calc_(query, VectorById(entry_id));
  Candidate entry{entry_id, entry_dist};
  candidates.push(entry);
  results.push(entry);
  visited[entry_id] = 1;

  while (!candidates.empty()) {
    Candidate current = candidates.top();
    if (results.size() >= ef && current.distance > results.top().distance) {
      break;
    }
    candidates.pop();

    const auto& neighbors = nodes_[current.id].links[level];
    for (size_t neighbor : neighbors) {
      if (visited[neighbor]) {
        continue;
      }
      visited[neighbor] = 1;
      float dist = dist_calc_(query, VectorById(neighbor));
      if (results.size() < ef || dist < results.top().distance) {
        Candidate cand{neighbor, dist};
        candidates.push(cand);
        results.push(cand);
        if (results.size() > ef) {
          results.pop();
        }
      }
    }
  }

  std::vector<Candidate> out;
  out.reserve(results.size());
  while (!results.empty()) {
    out.push_back(results.top());
    results.pop();
  }
  std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
    return a.distance < b.distance;
  });
  return out;
}

std::vector<size_t> HnswIndex::SelectNeighborsHeuristic(const std::vector<Candidate>& candidates,
                                                        size_t max_neighbors) const {
  std::vector<size_t> selected;
  selected.reserve(max_neighbors);

  for (const auto& candidate : candidates) {
    bool good = true;
    for (size_t chosen_id : selected) {
      float dist_between = dist_calc_(VectorById(candidate.id), VectorById(chosen_id));
      if (dist_between < candidate.distance) {
        good = false;
        break;
      }
    }
    if (good) {
      selected.push_back(candidate.id);
      if (selected.size() >= max_neighbors) {
        return selected;
      }
    }
  }

  for (const auto& candidate : candidates) {
    if (selected.size() >= max_neighbors) {
      break;
    }
    if (std::find(selected.begin(), selected.end(), candidate.id) == selected.end()) {
      selected.push_back(candidate.id);
    }
  }

  return selected;
}

std::vector<size_t> HnswIndex::SelectNeighborsHeuristicFromNode(size_t base_id,
                                                                const std::vector<size_t>& candidates,
                                                                size_t max_neighbors) const {
  std::vector<Candidate> scored;
  scored.reserve(candidates.size());
  VectorView base = VectorById(base_id);

  for (size_t candidate_id : candidates) {
    scored.push_back({candidate_id, dist_calc_(base, VectorById(candidate_id))});
  }
  std::sort(scored.begin(), scored.end(), [](const Candidate& a, const Candidate& b) {
    return a.distance < b.distance;
  });

  std::vector<size_t> selected;
  selected.reserve(max_neighbors);
  for (const auto& candidate : scored) {
    bool good = true;
    for (size_t chosen_id : selected) {
      float dist_between = dist_calc_(VectorById(candidate.id), VectorById(chosen_id));
      if (dist_between < candidate.distance) {
        good = false;
        break;
      }
    }
    if (good) {
      selected.push_back(candidate.id);
      if (selected.size() >= max_neighbors) {
        return selected;
      }
    }
  }

  for (const auto& candidate : scored) {
    if (selected.size() >= max_neighbors) {
      break;
    }
    if (std::find(selected.begin(), selected.end(), candidate.id) == selected.end()) {
      selected.push_back(candidate.id);
    }
  }

  return selected;
}

void HnswIndex::LinkBidirectional(size_t source_id, size_t target_id, size_t level, size_t max_neighbors) {
  auto& source_links = nodes_[source_id].links[level];
  source_links.push_back(target_id);
  if (source_links.size() > max_neighbors) {
    source_links = SelectNeighborsHeuristicFromNode(source_id, source_links, max_neighbors);
  }
}

void HnswIndex::AddVector(LabelType label, VectorView vector) {
  if (vector.size() != dim_) {
    throw std::invalid_argument("Vector dimension mismatch");
  }

  std::unique_lock lock(rw_lock_);

  size_t old_data_size = data_.size();
  size_t old_label_size = labels_.size();
  try {
    data_.insert(data_.end(), vector.begin(), vector.end());
    labels_.push_back(label);
  } catch (...) {
    data_.resize(old_data_size);
    labels_.resize(old_label_size);
    throw;
  }

  size_t new_id = labels_.size() - 1;
  size_t level = RandomLevel();
  Node node;
  node.level = level;
  node.links.resize(level + 1);
  try {
    nodes_.push_back(std::move(node));
  } catch (...) {
    data_.resize(old_data_size);
    labels_.resize(old_label_size);
    throw;
  }

  if (count_.load(std::memory_order_relaxed) == 0) {
    entry_point_ = new_id;
    max_level_ = level;
    count_.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  size_t entry_point = entry_point_;

  for (size_t current_level = max_level_; current_level > level; --current_level) {
    entry_point = GreedySearchLevel(vector, entry_point, current_level);
  }

  size_t level_to_connect = std::min(level, max_level_);
  for (size_t current_level = level_to_connect + 1; current_level-- > 0;) {
    size_t max_neighbors = (current_level == 0) ? config_.M_max0 : config_.M;
    auto candidates = SearchLayer(vector, entry_point, config_.ef_construction, current_level);
    auto selected = SelectNeighborsHeuristic(candidates, max_neighbors);

    for (size_t neighbor_id : selected) {
      nodes_[new_id].links[current_level].push_back(neighbor_id);
      LinkBidirectional(neighbor_id, new_id, current_level, max_neighbors);
    }

    if (!candidates.empty()) {
      entry_point = candidates.front().id;
    }
  }

  if (level > max_level_) {
    entry_point_ = new_id;
    max_level_ = level;
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

  std::shared_lock lock(rw_lock_);
  if (count_.load(std::memory_order_relaxed) == 0) {
    return {};
  }

  size_t entry_point = entry_point_;
  for (size_t current_level = max_level_; current_level > 0; --current_level) {
    entry_point = GreedySearchLevel(query, entry_point, current_level);
  }

  size_t ef = std::max(config_.ef_search, k);
  auto candidates = SearchLayer(query, entry_point, ef, 0);

  std::vector<SearchResult> results;
  results.reserve(std::min(k, candidates.size()));
  for (size_t i = 0; i < candidates.size() && results.size() < k; ++i) {
    results.push_back({labels_[candidates[i].id], candidates[i].distance});
  }

  return results;
}

} // namespace vectordb
