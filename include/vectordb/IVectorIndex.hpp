#pragma once

#include "Types.hpp"
#include <vector>

namespace vectordb {

/**
 * @brief Abstract interface for vector indices.
 *
 * All indices must implement this interface to allow for interchangeable use.
 */
class IVectorIndex {
 public:
  virtual ~IVectorIndex() = default;

  /**
   * @brief Adds a vector to the index.
   *
   * @param label Unique identifier for the vector.
   * @param vector The vector data.
   * @throw std::invalid_argument if vector dimension doesn't match index dimension.
   */
  virtual void AddVector(LabelType label, VectorView vector) = 0;

  /**
   * @brief Searches for the k-nearest neighbors of a query vector.
   *
   * @param query The query vector.
   * @param k Number of neighbors to return.
   * @return std::vector<SearchResult> Sorted list of nearest neighbors.
   */
  [[nodiscard]] virtual std::vector<SearchResult> Search(VectorView query, size_t k) const = 0;

  /**
   * @brief Returns the number of vectors in the index.
   */
  [[nodiscard]] virtual size_t Size() const noexcept = 0;

  /**
   * @brief Returns the dimension of vectors in the index.
   */
  [[nodiscard]] virtual size_t Dimension() const noexcept = 0;
};

} // namespace vectordb
