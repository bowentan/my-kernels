alias s := sync-runpod
alias ie := init-env
alias b := build

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

test-python python_bin="python":
  {{python_bin}} -m pytest tests/python

test-cpp:
  ctest --test-dir build --output-on-failure

bench-cpp args="":
  ./build/benchmarks/cpp/bench_add {{args}}
