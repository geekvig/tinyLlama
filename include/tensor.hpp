#pragma once

#include <vector>
#include <cstddef>
#include <sstream>
#include <numeric>
#include <stdexcept>


namespace tinyml {

class Tensor { 
public:
    Tensor() = default;

    explicit Tensor(std::vector<int> shape): shape_(std::move(shape)) {
        recompute_strides();
        data_.assign(numel(), 0.0f);
    }

    Tensor(std::vector<int> shape, std::vector<float> data)
        : shape_(std::move(shape)), data_(std::move(data)) {
        recompute_strides();
        if ((int)data_.size() != numel()) {
            throw std::runtime_error("Tensor: data size does not match shape");
        }
    }

    int numel() const {
        int n = 1;
        for (int s : shape_) n *= s;
        return n;
    }

    // ---- shape / size ----
    const std::vector<int>& shape() const { return shape_; }
    int ndim() const { return (int)shape_.size(); }
    int dim(int i) const { return shape_[i]; }

    // ---- raw data access ----
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Flat (linear) indexing into the contiguous buffer.
    float& at_flat(int idx) { return data_[idx]; }
    float at_flat(int idx) const { return data_[idx]; }

    // Multi-index access, e.g. t.at({b, i, j}) for a 3D tensor.
    float& at(const std::vector<int>& idx) { return data_[flat_index(idx)]; }
    float at(const std::vector<int>& idx) const { return data_[flat_index(idx)]; }

    // ---- convenience for the very common 1D/2D cases ----
    float& at(int i) { return data_[i]; }
    float& at(int i, int j) { return data_[i * strides_[0] + j * strides_[1]]; }
    float at(int i, int j) const { return data_[i * strides_[0] + j * strides_[1]]; }

    
private:
    std::vector<float> data_;
    std::vector<int>   shape_;
    std::vector<int>   strides_;

    void recompute_strides() {
        strides_.assign(shape_.size(), 1);
        for (int i = (int)shape_.size() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    int flat_index(const std::vector<int>& idx) const {
        if (idx.size() != shape_.size()) {
            throw std::runtime_error("Tensor::at: index rank mismatch");
        }
        int flat = 0;
        for (size_t i = 0; i < idx.size(); ++i) flat += idx[i] * strides_[i];
        return flat;
    }


};
} // namespace tinyML