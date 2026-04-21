#include "my_kernels/add.h"
#include "support/exception_assert.h"
#include "support/tensor_assert.h"

#include <gtest/gtest.h>

TEST(AddCpuTest, DispatchMatchesTorchAdd) {
    torch::manual_seed(0);

    auto a = torch::randn({2, 7, 11}, torch::TensorOptions().dtype(torch::kFloat32));
    auto b = torch::randn({2, 7, 11}, torch::TensorOptions().dtype(torch::kFloat32));

    const auto actual = add(a, b);
    const auto expected = torch::add(a, b);

    my_kernels::test::assert_close(actual, expected);
    EXPECT_TRUE(actual.device().is_cpu());
}

TEST(AddCpuTest, KernelHandlesNonContiguousInputs) {
    torch::manual_seed(0);

    const auto a =
        torch::randn({64}, torch::TensorOptions().dtype(torch::kFloat64)).slice(0, 0, 64, 2);
    const auto b =
        torch::randn({64}, torch::TensorOptions().dtype(torch::kFloat64)).slice(0, 0, 64, 2);

    ASSERT_FALSE(a.is_contiguous());
    ASSERT_FALSE(b.is_contiguous());

    const auto actual = add_cpu(a, b);
    const auto expected = torch::add(a, b);

    my_kernels::test::assert_close(actual, expected, 1e-10, 1e-12);
    EXPECT_TRUE(actual.is_contiguous());
}

TEST(AddCpuTest, RejectsMismatchedShapes) {
    const auto a = torch::randn({8}, torch::TensorOptions().dtype(torch::kFloat32));
    const auto b = torch::randn({10}, torch::TensorOptions().dtype(torch::kFloat32));

    my_kernels::test::expect_throws_with([&] { static_cast<void>(add(a, b)); }, "same shape");
}

TEST(AddCpuTest, RejectsIntegerInputs) {
    const auto a = torch::ones({8}, torch::TensorOptions().dtype(torch::kInt32));
    const auto b = torch::ones({8}, torch::TensorOptions().dtype(torch::kInt32));

    my_kernels::test::expect_throws_with(
        [&] { static_cast<void>(add(a, b)); }, "float32 or float64");
}
