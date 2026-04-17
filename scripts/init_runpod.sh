#!/usr/bin/env bash

apt update
apt install -y cmake rsync python3-dev

# just
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
