#include "tensor.hpp"
#include "thread_pool.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace tinyml {
namespace ops {

// Don't bother waking the pool for fewer than this many multiply-adds /
// elements: the wake-up latency would dominate.
constexpr int64_t kParallelWork = 1 << 16;

// out[i][j] = sum_k a[i][k] * b[k][j]
// a: [M, K], b: [K, N] -> out: [M, N]
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2 || a.dim(1) != b.dim(0)) {
        throw std::runtime_error("matmul: shape mismatch " + a.to_string() + " x " + b.to_string());
    }
    int M = a.dim(0), K = a.dim(1), N = b.dim(1);
    Tensor out({M, N});

    // Compute the sub-block out[i0:i1, j0:j1]. k stays the outer-of-j loop
    // so b is read row-contiguously (cache friendly for row-major b).
    auto kernel = [&](int i0, int i1, int j0, int j1) {
        for (int i = i0; i < i1; ++i) {
            for (int k = 0; k < K; ++k) {
                float aik = a.at(i, k);
                if (aik == 0.0f) continue;
                for (int j = j0; j < j1; ++j) {
                    out.at(i, j) += aik * b.at(k, j);
                }
            }
        }
    };

    if ((int64_t)M * K * N < kParallelWork) {
        kernel(0, M, 0, N);
    } else if (M >= N) {
        // Many rows (prefill): give each thread a band of rows.
        parallel_for(M, [&](int64_t r0, int64_t r1) { kernel((int)r0, (int)r1, 0, N); });
    } else {
        // Few rows, many columns (single-token decode, M=1, and the big
        // lm_head projection): split the COLUMNS instead. Chunks are >= 16
        // columns so two threads rarely share a cache line.
        parallel_for(N, [&](int64_t c0, int64_t c1) { kernel(0, M, (int)c0, (int)c1); }, 16);
    }
    return out;
}

// Transpose a 2D tensor: [rows, cols] -> [cols, rows].
inline Tensor transpose(const Tensor& x) {
    if (x.ndim() != 2) throw std::runtime_error("transpose: expected 2D tensor");
    int rows = x.dim(0), cols = x.dim(1);
    Tensor out({cols, rows});
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out.at(c, r) = x.at(r, c);
        }
    }
    return out;
}

// Elementwise add, same shape required.
inline Tensor add(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel()) throw std::runtime_error("add: size mismatch");
    Tensor out(a.shape());
    parallel_for(a.numel(), [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) out.at_flat((int)i) = a.at_flat((int)i) + b.at_flat((int)i);
    }, kParallelWork / 4);
    return out;
}

// Softmax over the last dimension of a 2D tensor [rows, cols].
inline Tensor softmax_rows(const Tensor& x) {
    int rows = x.dim(0), cols = x.dim(1);
    Tensor out(x.shape());
    // Rows are independent -> split rows across threads.
    int64_t min_rows = std::max<int64_t>(1, (kParallelWork / 4) / std::max(1, cols));
    parallel_for(rows, [&](int64_t r0, int64_t r1) {
        for (int r = (int)r0; r < (int)r1; ++r) {
            float maxv = -1e30f;
            for (int c = 0; c < cols; ++c) maxv = std::max(maxv, x.at(r, c));
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float e = std::exp(x.at(r, c) - maxv);
                out.at(r, c) = e;
                sum += e;
            }
            for (int c = 0; c < cols; ++c) out.at(r, c) /= sum;
        }
    }, min_rows);
    return out;
}

// RMSNorm over the last dim of a 2D tensor [rows, dim], scaled by weight[dim].
// This is the normalization Llama uses (simpler than LayerNorm: no mean
// subtraction, no bias).
inline Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps = 1e-5f) {
    int rows = x.dim(0), dim = x.dim(1);
    Tensor out(x.shape());
    int64_t min_rows = std::max<int64_t>(1, (kParallelWork / 4) / std::max(1, dim));
    parallel_for(rows, [&](int64_t r0, int64_t r1) {
        for (int r = (int)r0; r < (int)r1; ++r) {
            float ss = 0.0f;
            for (int c = 0; c < dim; ++c) ss += x.at(r, c) * x.at(r, c);
            float scale = 1.0f / std::sqrt(ss / dim + eps);
            for (int c = 0; c < dim; ++c) {
                out.at(r, c) = x.at(r, c) * scale * weight.at_flat(c);
            }
        }
    }, min_rows);
    return out;
}

// SiLU / swish activation: x * sigmoid(x). Used in Llama's SwiGLU MLP.
inline Tensor silu(const Tensor& x) {
    Tensor out(x.shape());
    parallel_for(x.numel(), [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) {
            float v = x.at_flat((int)i);
            out.at_flat((int)i) = v / (1.0f + std::exp(-v));
        }
    }, kParallelWork / 4);
    return out;
}

// Look up rows from an embedding table for a batch of token ids.
//   table: [vocab_size, dim], ids: token ids -> returns [ids.size(), dim]
inline Tensor embed(const Tensor& table, const std::vector<int>& ids) {
    int dim = table.dim(1);
    Tensor out({(int)ids.size(), dim});
    // Validate serially first so the exception comes from the calling thread.
    for (int id : ids) {
        if (id < 0 || id >= table.dim(0)) throw std::runtime_error("embed: token id out of range");
    }
    int64_t min_rows = std::max<int64_t>(1, (kParallelWork / 4) / std::max(1, dim));
    parallel_for((int64_t)ids.size(), [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i)
            for (int c = 0; c < dim; ++c) out.at((int)i, c) = table.at(ids[(size_t)i], c);
    }, min_rows);
    return out;
}

// Elementwise multiply, same shape.
inline Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel()) throw std::runtime_error("mul: size mismatch");
    Tensor out(a.shape());
    parallel_for(a.numel(), [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) out.at_flat((int)i) = a.at_flat((int)i) * b.at_flat((int)i);
    }, kParallelWork / 4);
    return out;
}

// Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) V
//   q: [n_q, d_k]   (queries -- e.g. the current tokens)
//   k: [n_k, d_k]   (keys    -- e.g. all tokens seen so far, incl. cache)
//   v: [n_k, d_v]   (values)
// returns: [n_q, d_v]
//
// If causal is true, query i is only allowed to attend to keys j <= i
// (assuming q and k are aligned so that position i in q corresponds to
// position i in k -- true for the "no KV-cache yet" case; with a cache,
// the caller offsets accordingly, see attention_causal_offset below).
inline Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal = false) {
    if (q.dim(1) != k.dim(1)) throw std::runtime_error("attention: Q/K dim mismatch");
    if (k.dim(0) != v.dim(0)) throw std::runtime_error("attention: K/V length mismatch");

    int n_q = q.dim(0), n_k = k.dim(0), d_k = q.dim(1);
    float scale = 1.0f / std::sqrt((float)d_k);

    Tensor kt = transpose(k);          // [d_k, n_k]
    Tensor scores = matmul(q, kt);     // [n_q, n_k]
    for (int i = 0; i < scores.numel(); ++i) scores.at_flat(i) *= scale;

    if (causal) {
        // mask out j > i (future keys), using the convention that key
        // index i is "the same position" as query index i, offset by
        // however many earlier (cached) keys precede this batch.
        int offset = n_k - n_q; // number of cached keys before this query block
        for (int i = 0; i < n_q; ++i) {
            for (int j = offset + i + 1; j < n_k; ++j) {
                scores.at(i, j) = -1e30f;
            }
        }
    }

    Tensor attn = softmax_rows(scores);   // [n_q, n_k]
    return matmul(attn, v);               // [n_q, d_v]
}

// Multi-head attention: splits q/k/v's feature dim into n_heads equal
// chunks, runs attention() independently per head, and concatenates the
// results back into [n_q, dim]. This is what a real transformer block uses
// (single-head attention above is the building block / teaching version).
inline Tensor multi_head_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                    int n_heads, bool causal = false) {
    int n_q = q.dim(0), n_k = k.dim(0), dim = q.dim(1);
    if (dim % n_heads != 0) throw std::runtime_error("multi_head_attention: dim not divisible by n_heads");
    int head_dim = dim / n_heads;

    Tensor out({n_q, dim});

    auto run_head = [&](int h) {
        // Slice out this head's columns from q, k, v.
        Tensor qh({n_q, head_dim}), kh({n_k, head_dim}), vh({n_k, head_dim});
        for (int i = 0; i < n_q; ++i)
            for (int c = 0; c < head_dim; ++c)
                qh.at(i, c) = q.at(i, h * head_dim + c);
        for (int i = 0; i < n_k; ++i)
            for (int c = 0; c < head_dim; ++c) {
                kh.at(i, c) = k.at(i, h * head_dim + c);
                vh.at(i, c) = v.at(i, h * head_dim + c);
            }

        Tensor oh = attention(qh, kh, vh, causal); // [n_q, head_dim]
        // Each head writes its own disjoint column band of `out`.
        for (int i = 0; i < n_q; ++i)
            for (int c = 0; c < head_dim; ++c)
                out.at(i, h * head_dim + c) = oh.at(i, c);
    };

    // Heads are fully independent -> one task per head. (The matmuls inside
    // attention() run inline here: nested parallel regions don't fan out.)
    int64_t work = (int64_t)n_heads * n_q * n_k * head_dim;
    if (work < kParallelWork) {
        for (int h = 0; h < n_heads; ++h) run_head(h);
    } else {
        parallel_for(n_heads, [&](int64_t h0, int64_t h1) {
            for (int64_t h = h0; h < h1; ++h) run_head((int)h);
        });
    }
    return out;
}

// Rotary position embeddings (RoPE), the position encoding Llama uses --
// applied to Q and K (not V) before the attention dot-product. Rotates
// each consecutive pair of dims (2i, 2i+1) by an angle that depends on
// the token's absolute position and the pair index.
//   x: [seq_len, dim], position_offset: absolute position of row 0
inline Tensor rope(const Tensor& x, int position_offset, float theta_base = 10000.0f) {
    int seq_len = x.dim(0), dim = x.dim(1);
    Tensor out(x.shape());
    // Each token position rotates independently -> split positions.
    int64_t min_rows = std::max<int64_t>(1, (kParallelWork / 4) / std::max(1, dim));
    parallel_for(seq_len, [&](int64_t p0, int64_t p1) {
        for (int pos_i = (int)p0; pos_i < (int)p1; ++pos_i) {
            int pos = position_offset + pos_i;
            for (int i = 0; i < dim / 2; ++i) {
                float freq = 1.0f / std::pow(theta_base, (2.0f * i) / dim);
                float angle = pos * freq;
                float cos_a = std::cos(angle), sin_a = std::sin(angle);
                float x0 = x.at(pos_i, 2 * i);
                float x1 = x.at(pos_i, 2 * i + 1);
                out.at(pos_i, 2 * i)     = x0 * cos_a - x1 * sin_a;
                out.at(pos_i, 2 * i + 1) = x0 * sin_a + x1 * cos_a;
            }
        }
    }, min_rows);
    return out;
}

} // namespace ops
} //