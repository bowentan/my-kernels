#include "my_kernels/add.h"
#include "support/tensor_assert.h"
#include "support/test_macros.h"

#include <torch/cuda.h>

namespace {

    int test_dispatch_matches_torch_add() {
        return my_kernels::test::run_test("add dispatch matches torch::add on CUDA", [] {
            torch::manual_seed(0);

            const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
            const auto a = torch::randn({1 << 20}, options);
            const auto b = torch::randn({1 << 20}, options);

            const auto actual = add(a, b);
            const auto expected = torch::add(a, b);

            my_kernels::test::assert_close(actual, expected);
            REQUIRE(actual.is_cuda());
        });
    }

    int test_cuda_kernel_handles_non_contiguous_inputs() {
        return my_kernels::test::run_test("add_cuda handles non-contiguous CUDA inputs", [] {
            torch::manual_seed(0);

            const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat64);
            const auto a = torch::randn({128}, options).slice(0, 0, 128, 2);
            const auto b = torch::randn({128}, options).slice(0, 0, 128, 2);

            REQUIRE(!a.is_contiguous());
            REQUIRE(!b.is_contiguous());

            const auto actual = add_cuda(a, b);
            const auto expected = torch::add(a, b);

            my_kernels::test::assert_close(actual, expected, 1e-10, 1e-12);
        });
    }

    int test_rejects_mixed_devices() {
        return my_kernels::test::run_test("add rejects mixed devices", [] {
            const auto a = torch::randn(
                {32}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
            const auto b = torch::randn(
                {32}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

            REQUIRE_THROWS_WITH(add(a, b), "same device");
        });
    }

} // namespace

int main() {
    if (!torch::cuda::is_available()) {
        std::cout << "[SKIP] CUDA runtime not available\n";
        return 0;
    }

    int failures = 0;
    failures += test_dispatch_matches_torch_add();
    failures += test_cuda_kernel_handles_non_contiguous_inputs();
    failures += test_rejects_mixed_devices();
    return failures == 0 ? 0 : 1;
}
