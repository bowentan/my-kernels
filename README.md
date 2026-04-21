> [!NOTE]
> This repo is only for learning and practice purpose.

# My Kernels

A self-own kernel library with CUDA and PyTorch integration for learning purpose.

## Included

- CPU implementation
- CUDA implementation
- C++ API
- PyTorch registration
- Python Wrapper
- Tests
  - Python correctness/dispatch/perf smoke
  - C++ unit-style native tests via CTest
- Benchmark against SOTA kernels
  - Python helpers
  - C++ executable benchmarks


## Repo structure

```txt
my-kernels/
├── README.md
├── pyproject.toml
├── CMakeLists.txt
├── include/
│   └── my_kernels/
├── benchmarks/
│   ├── cpp/
│   │   ├── add/
│   │   └── support/
│   └── python/
│       └── utils/
├── src/
│   ├── pytorch/
│   │   └── register_ops.cpp
│   └── ops/
├── tests/
│   ├── cpp/
│   │   ├── add/
│   │   └── support/
│   └── python/
│       ├── correctness/
│       ├── dispatch/
│       ├── perf_smoke/
│       └── reference/
├── python/
│   └── my_kernels/
│       ├── __init__.py
│       ├── _load_lib.py
│       └── ops.py
└── scripts/
```

## Build and run

```bash
just build python_bin=.venv/bin/python
just test-python python_bin=.venv/bin/python
just test-cpp
./build/benchmarks/cpp/bench_add --device cpu --size 1048576 --iters 100
```

Use the Python interpreter that already has `torch` installed when running `just build` or `just test-python`. Use `just build-cpp` when you only want the native library, tests, and benchmarks without the Python extension.

## Best-practice layout

- Keep implementation under `src/ops/<op>/` and public declarations under `include/my_kernels/`.
- Keep Python validation under `tests/python/<category>/` and import benchmark helpers from `benchmarks/python/`.
- Keep native validation under `tests/cpp/<op>/` and register each executable in `tests/cpp/CMakeLists.txt`.
- Keep native benchmarking under `benchmarks/cpp/<op>/` and report timings instead of asserting hard performance thresholds.
- Expose one public C++ entry point per op, then test that API in C++ and the registered op in Python.

## Adding a new kernel

For a new op such as `mul`, mirror the same slices:

1. Add `include/my_kernels/mul.h` plus `src/ops/mul/`.
2. Add the C++ dispatch entry point that routes to CPU/CUDA implementations.
3. Add Python correctness and dispatch tests under `tests/python/`.
4. Add native C++ tests under `tests/cpp/mul/`.
5. Add a native benchmark under `benchmarks/cpp/mul/`.
