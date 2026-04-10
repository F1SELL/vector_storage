#include <gtest/gtest.h>
#include "vectordb/FlatIndex.hpp"
#include "vectordb/HnswIndex.hpp"
#include <algorithm>
#include <random>

namespace vectordb::tests {

class IndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    data.resize(num_vectors, std::vector<float>(dim));
    for (auto& vec : data) {
      for (float& val : vec) val = dist(rng);
    }
  }

  const size_t dim = 16;
  const size_t num_vectors = 1000;
  std::vector<std::vector<float>> data;
};

TEST_F(IndexTest, FlatIndexExactMatch) {
  FlatIndex index(dim, MetricType::L2Squared);
  for (size_t i = 0; i < num_vectors; ++i) {
    index.AddVector(i, VectorView(data[i].data(), dim));
  }

  EXPECT_EQ(index.Size(), num_vectors);

  auto results = index.Search(VectorView(data[42].data(), dim), 5);
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(results[0].label, 42);
  EXPECT_NEAR(results[0].distance, 0.0f, 1e-6f);
}

TEST_F(IndexTest, FlatIndexReturnsSortedNearestNeighbors) {
  FlatIndex index(dim, MetricType::L2Squared);
  for (size_t i = 0; i < num_vectors; ++i) {
    index.AddVector(i, VectorView(data[i].data(), dim));
  }

  auto results = index.Search(VectorView(data[7].data(), dim), 10);
  ASSERT_EQ(results.size(), 10);
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i - 1].distance, results[i].distance);
  }
}

TEST_F(IndexTest, DimensionMismatchThrows) {
  FlatIndex index(dim, MetricType::L2Squared);
  std::vector<float> bad_vec(dim + 1, 0.0f);

  EXPECT_THROW(index.AddVector(1, VectorView(bad_vec.data(), bad_vec.size())), std::invalid_argument);
  EXPECT_THROW(
      {
        auto res = index.Search(VectorView(bad_vec.data(), bad_vec.size()), 1);
        (void)res;
      },
      std::invalid_argument);
}

TEST_F(IndexTest, HnswIsStubAndThrowsOnSearch) {
  HnswIndex index(dim, MetricType::L2Squared);
  for (size_t i = 0; i < 10; ++i) {
    index.AddVector(i, VectorView(data[i].data(), dim));
  }
  EXPECT_EQ(index.Size(), 10);
  EXPECT_THROW(
      {
        auto res = index.Search(VectorView(data[0].data(), dim), 5);
        (void)res;
      },
      std::logic_error);
}

} // namespace vectordb::tests