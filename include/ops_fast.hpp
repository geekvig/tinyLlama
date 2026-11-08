#pragma once

//
//   1. SIMD: instead of one float multiply-add per instruction, AVX2 does
//      8 float32 FMAs (fused multiply-add) per instruction. The trick is
//      giving the CPU contiguous memory to vectorize over -- which is
//      why we transpose B first, so each (row_i . row_j) dot product
//      reads two contiguous float arrays instead of striding through a
//      column.
//   2. Threading: split the output across a persistent thread pool
//      (thread_pool.hpp) instead of spawning std::threads per call.
//      Independent output rows have zero data dependency, so this is an
//      "embarrassingly parallel" split -- no locks needed.
//

#include "tensor.hpp"
#include "ops.hpp"
#include "thread_pool.hpp"
#include <thread>
#include <vector>
#include <algorithm>
#include <stdexcept>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define TINYML_HAVE_AVX2 1
#endif

namespace tinyml {
namespace ops {

// Dot product of two contiguous float arrays of length n.
inline float dot_scalar(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

#ifdef TINYML_HAVE_AVX2
inline float dot_simd(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc); // acc += va * vb, one instruction
    }
    // Horizontal sum of the 8 lanes in acc.
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sums = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float result = _mm_cvtss_f32(sums);

    // Tail elements that didn't fit in a group of 8.
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}
inline float dot(const float* a, const float* b, int n) { return dot_simd(a, b, n); }
#else
inline float dot(const float* a, const float* b, int n) { return dot_scalar(a, b, n); }
#endif

// Fast matmul: transpose B once, then split the M rows across the shared
// thread pool, each task computing its slice with the SIMD dot product.
// a: [M, K], b: [K, N] -> out: [M, N]  (same contract as ops::matmul).
//
// n_threads caps the split for THIS call (<=0 means "use the global
// setting from set_num_threads()"). When M is small (decode: M=1) the
// columns are split instead, so a single-token projection still uses
// every core.
inline Tensor matmul_fast(const Tensor& a, const Tensor& b, int n_threads = 0) {
    if (a.ndim() != 2 || b.ndim() != 2 || a.dim(1) != b.dim(0))
        throw std::runtime_error("matmul_fast: shape mismatch");
    int M = a.dim(0), K = a.dim(1), N = b.dim(1);

    Tensor b_t = transpose(b); // [N, K] -- pay this once, reuse for every row
    Tensor out({M, N});

    auto kernel = [&](int i0, int i1, int j0, int j1) {
        for (int i = i0; i < i1; ++i) {
            const float* a_row = a.data() + (size_t)i * K;
            for (int j = j0; j < j1; ++j) {
                out.at(i, j) = dot(a_row, b_t.data() + (size_t)j * K, K);
            }
        }
    };

    if (M >= N) {
        parallel_for(M, [&](int64_t r0, int64_t r1) { kernel((int)r0, (int)r1, 0, N); }, 1, n_threads);
    } else {
        parallel_for(N, [&](int64_t c0, int64_t c1) { kernel(0, M, (int)c0, (int)c1); }, 16, n_threads);
    }
    return out;
}

} // namespace ops
} // namespace tinyml
