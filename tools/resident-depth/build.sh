#!/usr/bin/env bash
# Resolve native dependencies once before starting a resident round.
set -euo pipefail
here=$(dirname "$(readlink -f "$0")")
output="$here/../../Intermediate/resident-depth"
mkdir -p "$output"
: "${EVAL_COMPILER:?Source object-generator/env.sh first}"
: "${EVAL_CUDA_RUNTIME:?Source object-generator/env.sh first}"
headers=${EVAL_VULKAN_HEADERS:-$(nix build --no-link --print-out-paths nixpkgs#vulkan-headers)}
libpng=${EVAL_LIBPNG:-$(nix build --no-link --print-out-paths nixpkgs#libpng)}
png_headers=${EVAL_PNG_HEADERS:-$(nix build --no-link --print-out-paths nixpkgs#libpng.dev)}
loader=${EVAL_VULKAN_LOADER:-$(nix build --no-link --print-out-paths nixpkgs#vulkan-loader)}
cargo=${EVAL_CARGO:-$(nix build --no-link --print-out-paths nixpkgs#cargo)/bin/cargo}
rustc=${EVAL_RUSTC:-$(nix build --no-link --print-out-paths nixpkgs#rustc)/bin/rustc}
export PATH="$EVAL_COMPILER/bin:$(dirname "$rustc"):$PATH"
export RUSTC="$rustc" CARGO_TARGET_DIR="$output/shader-compiler"
"$cargo" build --locked --manifest-path "$here/shader-compiler/Cargo.toml"
# Vulkan structs deliberately zero-initialize unspecified optional fields.
"$EVAL_COMPILER/bin/g++" -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -fPIC -shared -I"$headers/include" -I"$EVAL_CUDA_RUNTIME/include" -I"$png_headers/include" \
 "$here/vulkan_backend.cpp" "$loader/lib/libvulkan.so.1" "$libpng/lib/libpng16.so.16" /run/opengl-driver/lib/libcuda.so.1 -o "$output/resident_vk.so.new"
mv "$output/resident_vk.so.new" "$output/resident_vk.so"
python3 "$here/compile_shaders.py" "$output" "$CARGO_TARGET_DIR/debug/object-shader-compile"
