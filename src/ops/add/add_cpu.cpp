#include "my_kernels/add.h"

#include <ATen/Parallel.h>

torch::Tensor add_cpu(torch::Tensor a, torch::Tensor b) {
    check_inputs(a, b);
    TORCH_CHECK(a.device().is_cpu(), "CPU tensors expected");

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto out = torch::empty_like(a_c);
    const int64_t n = a_c.numel();

    AT_DISPATCH_FLOATING_TYPES(a_c.scalar_type(), "add_cpu", [&] {
        const auto* a_ptr = a_c.data_ptr<scalar_t>();
        const auto* b_ptr = b_c.data_ptr<scalar_t>();
        auto* out_ptr = out.data_ptr<scalar_t>();

        at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
            for (int64_t i{begin}; i < end; ++i) {
                out_ptr[i] = a_ptr[i] + b_ptr[i];
            }
        });
    });

    return out;
}
