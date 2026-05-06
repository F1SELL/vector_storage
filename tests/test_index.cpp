#include <gtest/gtest.h>
#include "vectordb/FlatIndex.hpp"
#include "vectordb/HnswIndex.hpp"
#include <algorithm>
#include <random>
#include <unordered_set>

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

TEST_F(IndexTest, HnswExactMatchIsTop1) {
  HnswIndex::Config config;
  config.ef_search = 100;
  config.ef_construction = 200;
  HnswIndex index(dim, MetricType::L2Squared, config);

  for (size_t i = 0; i < num_vectors; ++i) {
    index.AddVector(i, VectorView(data[i].data(), dim));
  }

  auto results = index.Search(VectorView(data[123].data(), dim), 1);
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].label, 123);
}

TEST_F(IndexTest, HnswRecallVsFlatIsHighOnRandomData) {
  HnswIndex::Config config;
  config.ef_search = 100;
  config.ef_construction = 200;
  HnswIndex hnsw(dim, MetricType::L2Squared, config);
  FlatIndex flat(dim, MetricType::L2Squared);

  for (size_t i = 0; i < num_vectors; ++i) {
    hnsw.AddVector(i, VectorView(data[i].data(), dim));
    flat.AddVector(i, VectorView(data[i].data(), dim));
  }

  const size_t k = 10;
  const size_t queries = 100;
  size_t hit_count = 0;

  for (size_t i = 0; i < queries; ++i) {
    size_t qid = (i * 17) % num_vectors;
    auto exact = flat.Search(VectorView(data[qid].data(), dim), k);
    auto approx = hnsw.Search(VectorView(data[qid].data(), dim), k);

    std::unordered_set<LabelType> exact_labels;
    for (const auto& res : exact) {
      exact_labels.insert(res.label);
    }
    for (const auto& res : approx) {
      if (exact_labels.contains(res.label)) {
        ++hit_count;
      }
    }
  }

  double recall = static_cast<double>(hit_count) / static_cast<double>(queries * k);
  EXPECT_GE(recall, 0.90);
}

TEST_F(IndexTest, HnswHandlesEmptyAndKZero) {
  HnswIndex index(dim, MetricType::L2Squared);
  auto empty = index.Search(VectorView(data[0].data(), dim), 10);
  EXPECT_TRUE(empty.empty());

  for (size_t i = 0; i < 10; ++i) {
    index.AddVector(i, VectorView(data[i].data(), dim));
  }

  auto zero_k = index.Search(VectorView(data[0].data(), dim), 0);
  EXPECT_TRUE(zero_k.empty());
}

TEST_F(IndexTest, HnswDimensionMismatchThrows) {
  HnswIndex index(dim, MetricType::L2Squared);
  std::vector<float> bad_vec(dim + 1, 0.0f);

  EXPECT_THROW(index.AddVector(1, VectorView(bad_vec.data(), bad_vec.size())), std::invalid_argument);
  EXPECT_THROW(
      {
        auto res = index.Search(VectorView(bad_vec.data(), bad_vec.size()), 1);
        (void)res;
      },
      std::invalid_argument);
}

} // namespace vectordb::tests