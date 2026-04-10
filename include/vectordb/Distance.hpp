#pragma once

#include "Types.hpp"
#include <cmath>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__ARM_NEON) && !defined(VECTORDB_DISABLE_NEON)
#include <arm_neon.h>
#endif

namespace vectordb {

enum class MetricType {
  L2Squared,
  DotProduct,
  Cosine
};


inline float L2SquaredScalar(VectorView a, VectorView b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

inline float DotProductScalar(VectorView a, VectorView b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return -sum;
}

inline float CosineScalar(VectorView a, VectorView b) {
  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;
  return 1.0f - (dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}


#if defined(__AVX512F__)

inline float L2SquaredAVX512(VectorView a, VectorView b) {
    size_t n = a.size();
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }
    float res = _mm512_reduce_add_ps(sum);
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        res += diff * diff;
    }
    return res;
}

inline float DotProductAVX512(VectorView a, VectorView b) {
    size_t n = a.size();
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    float res = _mm512_reduce_add_ps(sum);
    for (; i < n; ++i) res += a[i] * b[i];
    return -res;
}

#elif defined(__ARM_NEON) && !defined(VECTORDB_DISABLE_NEON)

inline float L2SquaredNEON(VectorView a, VectorView b) {
  size_t n = a.size();
  float32x4_t sum = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 3 < n; i += 4) {
    float32x4_t va = vld1q_f32(&a[i]);
    float32x4_t vb = vld1q_f32(&b[i]);
    float32x4_t diff = vsubq_f32(va, vb);
    sum = vmlaq_f32(sum, diff, diff);
  }
  float res = vaddvq_f32(sum);
  for (; i < n; ++i) {
    float diff = a[i] - b[i];
    res += diff * diff;
  }
  return res;
}

inline float DotProductNEON(VectorView a, VectorView b) {
  size_t n = a.size();
  float32x4_t sum = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 3 < n; i += 4) {
    float32x4_t va = vld1q_f32(&a[i]);
    float32x4_t vb = vld1q_f32(&b[i]);
    sum = vmlaq_f32(sum, va, vb);
  }
  float res = vaddvq_f32(sum);
  for (; i < n; ++i) res += a[i] * b[i];
  return -res;
}

#endif

class DistanceCalculator {
 public:
  explicit DistanceCalculator(size_t dim, MetricType metric)
      : dim_(dim), metric_(metric) {}

  float operator()(VectorView a, VectorView b) const {
    if (a.size() != dim_ || b.size() != dim_) {
      throw std::invalid_argument("Vector dimension mismatch");
    }

    switch (metric_) {
      case MetricType::L2Squared:
#if defined(__AVX512F__)
        return L2SquaredAVX512(a, b);
#elif defined(__ARM_NEON) && !defined(VECTORDB_DISABLE_NEON)
        return L2SquaredNEON(a, b);
#else
        return L2SquaredScalar(a, b);
#endif
      case MetricType::DotProduct:
#if defined(__AVX512F__)
        return DotProductAVX512(a, b);
#elif defined(__ARM_NEON) && !defined(VECTORDB_DISABLE_NEON)
        return DotProductNEON(a, b);
#else
        return DotProductScalar(a, b);
#endif
      case MetricType::Cosine:
        return CosineScalar(a, b);
      default:
        return L2SquaredScalar(a, b);
    }
  }

 private:
  size_t dim_;
  MetricType metric_;
};

} // namespace vectordb
