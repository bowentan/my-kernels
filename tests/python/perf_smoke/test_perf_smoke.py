import torch

from benchmarks.python.utils.timer import bench
import my_kernels


def test_add_perf_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    a = torch.randn(1 << 20, device=device, dtype=torch.float32)
    b = torch.randn(1 << 20, device=device, dtype=torch.float32)

    ref = bench(torch.add, a, b, warmup=5, iters=20)
    stats = bench(my_kernels.add, a, b, warmup=5, iters=20)

    assert stats["median_ms"] <= ref["median_ms"] * 1.2
