import statistics
import time

import torch


def _uses_cuda(args) -> bool:
    return any(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in args)


def bench(fn, *args, warmup=10, iters=100):
    if _uses_cuda(args):
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_ms = []

        for _ in range(iters):
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    else:
        for _ in range(warmup):
            fn(*args)

        times_ms = []
        for _ in range(iters):
            start = time.perf_counter()
            fn(*args)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000)

    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }
