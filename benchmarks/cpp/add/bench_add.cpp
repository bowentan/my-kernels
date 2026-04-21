#include "my_kernels/add.h"
#include "support/benchmark.h"

#include <iostream>
#include <string>
#include <string_view>
#include <torch/cuda.h>

namespace {

    struct Options {
        std::string device = torch::cuda::is_available() ? "cuda" : "cpu";
        int64_t size = 1 << 20;
        int warmup = 10;
        int iters = 50;
        torch::ScalarType dtype = torch::kFloat32;
    };

    torch::ScalarType parse_dtype(std::string_view value) {
        if (value == "float32") {
            return torch::kFloat32;
        }
        if (value == "float64") {
            return torch::kFloat64;
        }

        throw std::runtime_error("unsupported dtype, expected float32 or float64");
    }

    Options parse_args(int argc, char** argv) {
        Options options;

        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];

            if (arg == "--device" && i + 1 < argc) {
                options.device = argv[++i];
                continue;
            }

            if (arg == "--size" && i + 1 < argc) {
                options.size = std::stoll(argv[++i]);
                continue;
            }

            if (arg == "--warmup" && i + 1 < argc) {
                options.warmup = std::stoi(argv[++i]);
                continue;
            }

            if (arg == "--iters" && i + 1 < argc) {
                options.iters = std::stoi(argv[++i]);
                continue;
            }

            if (arg == "--dtype" && i + 1 < argc) {
                options.dtype = parse_dtype(argv[++i]);
                continue;
            }

            throw std::runtime_error("unknown argument: " + std::string(arg));
        }

        return options;
    }

    void print_result(std::string_view label,
                      const my_kernels::benchmark::BenchmarkResult& result) {
        std::cout << label << ": median=" << result.median_ms << " ms"
                  << ", min=" << result.min_ms << " ms"
                  << ", max=" << result.max_ms << " ms\n";
    }

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        if (options.device == "cuda" && !torch::cuda::is_available()) {
            throw std::runtime_error("CUDA benchmark requested, but CUDA is not available");
        }

        const auto tensor_options =
            torch::TensorOptions()
                .device(options.device == "cuda" ? torch::kCUDA : torch::kCPU)
                .dtype(options.dtype);

        const auto a = torch::randn({options.size}, tensor_options);
        const auto b = torch::randn({options.size}, tensor_options);

        const auto reference = torch::add(a, b);
        const auto candidate = add(a, b);

        if (!at::allclose(candidate, reference, 1e-5, 1e-8, true)) {
            throw std::runtime_error("benchmark sanity check failed: outputs do not match");
        }

        const auto torch_stats = my_kernels::benchmark::run([&] { return torch::add(a, b); },
                                                            options.warmup, options.iters);

        const auto my_kernel_stats =
            my_kernels::benchmark::run([&] { return add(a, b); }, options.warmup, options.iters);

        std::cout << "device=" << options.device << " size=" << options.size
                  << " dtype=" << (options.dtype == torch::kFloat32 ? "float32" : "float64")
                  << " warmup=" << options.warmup << " iters=" << options.iters << '\n';

        print_result("torch::add", torch_stats);
        print_result("my_kernels::add", my_kernel_stats);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "benchmark failed: " << ex.what() << '\n';
        return 1;
    }
}
