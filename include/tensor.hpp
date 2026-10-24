#pragma once

#include <vector>
#include <cstddef>
#include <sstream>
#include <numeric>
#include <stdexcept>

namespace tinyML {

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



};
} // namespace tinyML