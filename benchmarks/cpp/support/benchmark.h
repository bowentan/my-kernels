#pragma once

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <vector>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace my_kernels::benchmark {

    struct BenchmarkResult {
        double min_ms{};
        double median_ms{};
        double max_ms{};
    };

    inline void synchronize(const torch::Tensor& tensor) {
#ifdef WITH_CUDA
        if (tensor.is_cuda()) {
            const auto status = cudaDeviceSynchronize();
            if (status != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(status));
            }
        }
#else
        static_cast<void>(tensor);
#endif
    }

    template <typename Fn> inline BenchmarkResult run(Fn&& fn, int warmup, int iters) {
        if (warmup < 0 || iters <= 0) {
            throw std::runtime_error("warmup must be >= 0 and iters must be > 0");
        }

        for (int i = 0; i < warmup; ++i) {
            synchronize(fn());
        }

        std::vector<double> samples_ms;
        samples_ms.reserve(iters);

        for (int i = 0; i < iters; ++i) {
            const auto start = std::chrono::steady_clock::now();
            const auto output = fn();
            synchronize(output);
            const auto end = std::chrono::steady_clock::now();

            const auto elapsed = std::chrono::duration<double, std::milli>(end - start);
            samples_ms.push_back(elapsed.count());
        }

        std::sort(samples_ms.begin(), samples_ms.end());

        BenchmarkResult result;
        result.min_ms = samples_ms.front();
        result.median_ms = samples_ms[samples_ms.size() / 2];
        result.max_ms = samples_ms.back();
        return result;
    }

} // namespace my_kernels::benchmark
