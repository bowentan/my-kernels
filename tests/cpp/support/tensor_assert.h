#pragma once

#include <sstream>
#include <stdexcept>
#include <torch/torch.h>

namespace my_kernels::test {

    inline void assert_close(const torch::Tensor& actual, const torch::Tensor& expected,
                             double rtol = 1e-5, double atol = 1e-8) {
        if (!actual.sizes().equals(expected.sizes())) {
            throw std::runtime_error("tensor shapes do not match");
        }

        if (actual.scalar_type() != expected.scalar_type()) {
            throw std::runtime_error("tensor dtypes do not match");
        }

        if (actual.device() != expected.device()) {
            throw std::runtime_error("tensor devices do not match");
        }

        if (at::allclose(actual, expected, rtol, atol, true)) {
            return;
        }

        const double max_diff =
            actual.numel() == 0 ? 0.0 : (actual - expected).abs().max().item<double>();

        std::ostringstream message;
        message << "tensors are not close (rtol=" << rtol << ", atol=" << atol
                << ", max_diff=" << max_diff << ")";
        throw std::runtime_error(message.str());
    }

} // namespace my_kernels::test
