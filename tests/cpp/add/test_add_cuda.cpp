#include "my_kernels/add.h"
#include "support/exception_assert.h"
#include "support/tensor_assert.h"

#include <gtest/gtest.h>
#include <torch/cuda.h>

namespace {

void require_cuda_runtime() {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA runtime not available";
    }
}

} // namespace

TEST(AddCudaTest, DispatchMatchesTorchAdd) {
    require_cuda_runtime();
    torch::manual_seed(0);

    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const auto a = torch::randn({1 << 20}, options);
    const auto b = torch::randn({1 << 20}, options);

    const auto actual = add(a, b);
    const auto expected = torch::add(a, b);

    my_kernels::test::assert_close(actual, expected);
    EXPECT_TRUE(actual.is_cuda());
}

TEST(AddCudaTest, KernelHandlesNonContiguousInputs) {
    require_cuda_runtime();
    torch::manual_seed(0);

    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat64);
    const auto a = torch::randn({128}, options).slice(0, 0, 128, 2);
    const auto b = torch::randn({128}, options).slice(0, 0, 128, 2);

    ASSERT_FALSE(a.is_contiguous());
    ASSERT_FALSE(b.is_contiguous());

    const auto actual = add_cuda(a, b);
    const auto expected = torch::add(a, b);

    my_kernels::test::assert_close(actual, expected, 1e-10, 1e-12);
}

TEST(AddCudaTest, RejectsMixedDevices) {
    require_cuda_runtime();

    const auto a =
        torch::randn({32}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    const auto b =
        torch::randn({32}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    my_kernels::test::expect_throws_with([&] { static_cast<void>(add(a, b)); }, "same device");
}
