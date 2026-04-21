#pragma once

#include <torch/torch.h>

inline void check_inputs(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.defined(), "a must be defined");
    TORCH_CHECK(b.defined(), "b must be defined");

    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same data type");
    TORCH_CHECK(a.dim() == b.dim(), "a and b must have the same shape");

    for (int64_t i{}; i < a.dim(); ++i) {
        TORCH_CHECK(a.size(i) == b.size(i), "a and b must have the same shape");
    }

    TORCH_CHECK(a.scalar_type() == at::kFloat || a.scalar_type() == at::kDouble,
                "a and b must be float32 or float64 tensors");
}

torch::Tensor add(torch::Tensor a, torch::Tensor b);
torch::Tensor add_cpu(torch::Tensor a, torch::Tensor b);
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
