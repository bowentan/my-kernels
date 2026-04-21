alias s := sync-runpod
alias ie := init-env
alias b := build
alias t := test

[positional-arguments]
sync-runpod addr dest_dir="/workspace/my-kernels":
  #!/usr/bin/env bash
  IFS=: read -r ip port <<< "{{addr}}"
  SSH_PORT=$port bash ./scripts/rsync_runpod.sh . root@${ip}:{{dest_dir}}

init-env:
  uv venv --python $(which python) --system-site-packages
  uv sync --no-install-project

build python_bin="python":
  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$({{python_bin}} -c "import torch; print(torch.utils.cmake_prefix_path)")

  cmake --build build --parallel

build-cpp python_bin="python":
  cmake -S . -B build \
    -DBUILD_PYTHON_EXTENSION=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$({{python_bin}} -c "import torch; print(torch.utils.cmake_prefix_path)")

  cmake --build build --parallel

test:
  ctest --test-dir build --output-on-failure

test-python:
  ctest --test-dir build --output-on-failure -L python

test-cpp:
  ctest --test-dir build --output-on-failure -L cpp

test-fast:
  ctest --test-dir build --output-on-failure -LE perf

bench-cpp args="":
  ./build/benchmarks/cpp/bench_add {{args}}
