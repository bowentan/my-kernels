#include "my_kernels/add.h"
#include "support/tensor_assert.h"
#include "support/test_macros.h"

namespace {

    int test_dispatch_matches_torch_add() {
        return my_kernels::test::run_test("add dispatch matches torch::add on CPU", [] {
            torch::manual_seed(0);

            auto a = torch::randn({2, 7, 11}, torch::TensorOptions().dtype(torch::kFloat32));
            auto b = torch::randn({2, 7, 11}, torch::TensorOptions().dtype(torch::kFloat32));

            const auto actual = add(a, b);
            const auto expected = torch::add(a, b);

            my_kernels::test::assert_close(actual, expected);
            REQUIRE(actual.device().is_cpu());
        });
    }

    int test_cpu_kernel_handles_non_contiguous_inputs() {
        return my_kernels::test::run_test("add_cpu handles non-contiguous CPU inputs", [] {
            torch::manual_seed(0);

            const auto a = torch::randn({64}, torch::TensorOptions().dtype(torch::kFloat64))
                               .slice(0, 0, 64, 2);
            const auto b = torch::randn({64}, torch::TensorOptions().dtype(torch::kFloat64))
                               .slice(0, 0, 64, 2);

            REQUIRE(!a.is_contiguous());
            REQUIRE(!b.is_contiguous());

            const auto actual = add_cpu(a, b);
            const auto expected = torch::add(a, b);

            my_kernels::test::assert_close(actual, expected, 1e-10, 1e-12);
            REQUIRE(actual.is_contiguous());
        });
    }

    int test_rejects_mismatched_shapes() {
        return my_kernels::test::run_test("add rejects mismatched shapes", [] {
            const auto a = torch::randn({8}, torch::TensorOptions().dtype(torch::kFloat32));
            const auto b = torch::randn({10}, torch::TensorOptions().dtype(torch::kFloat32));

            REQUIRE_THROWS_WITH(add(a, b), "same shape");
        });
    }

    int test_rejects_integer_inputs() {
        return my_kernels::test::run_test("add rejects integer tensors", [] {
            const auto a = torch::ones({8}, torch::TensorOptions().dtype(torch::kInt32));
            const auto b = torch::ones({8}, torch::TensorOptions().dtype(torch::kInt32));

            REQUIRE_THROWS_WITH(add(a, b), "float32 or float64");
        });
    }

} // namespace

int main() {
    int failures = 0;
    failures += test_dispatch_matches_torch_add();
    failures += test_cpu_kernel_handles_non_contiguous_inputs();
    failures += test_rejects_mismatched_shapes();
    failures += test_rejects_integer_inputs();
    return failures == 0 ? 0 : 1;
}
