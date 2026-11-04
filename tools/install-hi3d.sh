#!/usr/bin/env bash
# Builds Hi3D v1.0's own environment in its checkout: install-hi3d.sh <checkout>
# Its package pins PyTorch 2.5, whose kernels predate Blackwell, so it takes Espalier's PyTorch and flash-attn and installs
# its other dependencies around them.
set -euo pipefail
hi3d=$1
python=$hi3d/.venv/bin/python
abi=TRUE
requirements=$(sed -n '/^inference = \[/,/^\]/p;/^dependencies = \[/,/^\]/p' "$hi3d/pyproject.toml" | grep -oE '"[^"]+"' | tr -d '"' | grep -viE '^(torch|torchvision|flash-attn|flash_attn|spconv)')
[ -d "$hi3d/.venv" ] || uv venv --quiet --python 3.10 "$hi3d/.venv"
uv pip install --python "$python" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$python" packaging ninja psutil setuptools wheel spconv-cu126 $requirements \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.8cxx11abi$abi-cp310-cp310-linux_x86_64.whl"
uv pip install --python "$python" --no-deps -e "$hi3d"
