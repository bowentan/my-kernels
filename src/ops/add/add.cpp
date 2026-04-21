#include "my_kernels/add.h"

#include <utility>

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    check_inputs(a, b);

    if (a.is_cuda()) {
#ifdef WITH_CUDA
        return add_cuda(std::move(a), std::move(b));
#else
        TORCH_CHECK(false, "CUDA support is not enabled in this build");
#endif
    }

    return add_cpu(std::move(a), std::move(b));
}
