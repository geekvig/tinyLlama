#pragma once
// A minimal per-layer KV-cache: append new K/V rows as tokens are
// generated, and hand back the full K/V seen so far for attention.
// This is the O(1)-per-step trick that turns naive O(n^2) recomputation
// into the incremental decoding real inference engines use


#include "tensor.hpp"
#include <vector>

namespace tinyml {

class KVCache {
public:
    KVCache(int dim, int max_len) : dim_(dim), max_len_(max_len), len_(0) {
        k_.assign((size_t)max_len * dim, 0.0f);
        v_.assign((size_t)max_len * dim, 0.0f);
    }

    // Append new_k/new_v (each [n_new, dim]) to the cache.
    void append(const Tensor& new_k, const Tensor& new_v) {
        int n_new = new_k.dim(0);
        if (len_ + n_new > max_len_) throw std::runtime_error("KVCache: overflow");
        for (int i = 0; i < n_new; ++i) {
            for (int c = 0; c < dim_; ++c) {
                k_[(size_t)(len_ + i) * dim_ + c] = new_k.at(i, c);
                v_[(size_t)(len_ + i) * dim_ + c] = new_v.at(i, c);
            }
        }
        len_ += n_new;
    }

    // Return everything cached so far as [len_, dim] tensors.
    Tensor keys() const { return slice(k_); }
    Tensor values() const { return slice(v_); }

    int length() const { return len_; }

private:
    int dim_, max_len_, len_;
    std::vector<float> k_, v_;

    Tensor slice(const std::vector<float>& buf) const {
        std::vector<float> out(buf.begin(), buf.begin() + (size_t)len_ * dim_);
        return Tensor({len_, dim_}, out);
    }
};

} // namespace tinyml
