#pragma once

#include <cstdint>
#include <vector>
#include <span>

namespace vectordb {

/**
 * @brief Type for vector labels/identifiers.
 */
using LabelType = uint64_t;

/**
 * @brief Type for distance values.
 */
using DistanceType = float;

/**
 * @brief A view into a vector's data.
 *
 * Uses std::span for zero-copy access to contiguous memory.
 */
using VectorView = std::span<const float>;

/**
 * @brief Result of a nearest neighbor search.
 */
struct SearchResult {
  LabelType label;
  DistanceType distance;

  auto operator<=>(const SearchResult&) const = default;
};

} // namespace vectordb
