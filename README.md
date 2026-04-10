> [!INFO]
> This repo is only for learning and practice purpose.

# My CUDA Kernels

A self-own CUDA kernel library for learning purpose.

## Included

- CPU implementation
- CUDA implementation
- PyTorch registration
- Python Wrapper
- Tests
  - Correctness
  - Dispatch/Error
  - Perf smoke
- Benchmark against SOTA kernels


## Repo structure

```txt
my-kernels/
├── README.md
├── pyproject.toml
├── CMakeLists.txt
├── include/
│   └── my_kernels/
├── src/
│   ├── pytorch/
│   │   └── register_ops.cpp
│   └── ops/
├── python/
│   └── my_kernels/
│       ├── __init__.py
│       ├── _load_lib.py
│       └── reference/
│           ├── __init__.py
├── tests/
│   ├── conftest.py
│   ├── correctness/
│   ├── dispatch/
│   └── perf_smoke/
├── benchmarks/
│   ├── __init__.py
│   ├── baselines/
│   │   ├── __init__.py
│   ├── ops/
│   │   ├── __init__.py
│   └── utils/
│       ├── __init__.py
└── scripts/
```
