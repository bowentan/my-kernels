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

build:
  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

  cmake --build build --parallel
