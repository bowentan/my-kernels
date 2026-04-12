#!/usr/bin/env bash

apt update
apt install -y cmake nvtop rsync python3-dev

curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

pip install pytest ninja
