#include "my_kernels/add.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void add_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

torch::Tensor add_cuda(const torch::Tensor a, const torch::Tensor b) {
    check_inputs(a, b);
    TORCH_CHECK(a.is_cuda(), "CUDA tensor expected");

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto out = torch::empty_like(a_c);
    const int64_t n = a_c.numel();
    if (n == 0)
        return out;

    constexpr int nthreads = 256;
    const int nblock = static_cast<int>((n + nthreads - 1) / nthreads);

    AT_DISPATCH_FLOATING_TYPES(a_c.scalar_type(), "add_cuda", [&] {
        add_kernel<scalar_t><<<nblock, nthreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            a_c.data_ptr<scalar_t>(), b_c.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
