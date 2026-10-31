#!/usr/bin/env bash
# Compile CUDA C++ projection kernel into shared library.
set -euo pipefail
here=$(dirname "$(readlink -f "$0")")
output="$here/Intermediate/project_gpu.so"
[ ! "$here/project_gpu.cu" -nt "$output" ] && [ -f "$output" ] && exit 0
mkdir -p "$(dirname "$output")"

NVCC="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
if ! command -v "$NVCC" &>/dev/null; then
    NVCC=$(which nvcc 2>/dev/null || true)
fi
if [ -z "$NVCC" ] || [ ! -x "$NVCC" ]; then
    echo "Notice: nvcc not found; skipping project_gpu.cu compilation" >&2
    exit 0
fi

"$NVCC" -std=c++17 -O3 --shared -Xcompiler=-fPIC -Xcompiler=-Wall -Xcompiler=-Wextra \
 --fmad=false --allow-unsupported-compiler \
 "$here/project_gpu.cu" -o "$output.tmp.$$"
mv "$output.tmp.$$" "$output"
