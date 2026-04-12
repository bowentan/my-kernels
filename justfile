alias s := sync-runpod
alias b := build

[positional-arguments]
sync-runpod addr dest_dir="/workspace/my-kernels":
  #!/usr/bin/env bash
  IFS=: read -r ip port <<< "{{addr}}"
  SSH_PORT=$port bash ./scripts/rsync_runpod.sh . root@${ip}:{{dest_dir}}

build:
  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") \

  cmake --build build --parallel
