#!/usr/bin/env bash
# Build the Vulkan/CUDA resident baker, its WGSL shaders, and the GLB packager.
set -euo pipefail
root=$(dirname "$(dirname "$(readlink -f "$0")")")
output="$root/Intermediate"
mkdir -p "$output/resident-depth" "$output/surfaces"
: "${CXX:=c++}"
: "${CUDA_HOME:=$(dirname "$(dirname "$(command -v nvcc)")")}"
export CARGO_TARGET_DIR="$output/resident-depth/shader-compiler"
cargo build --locked --manifest-path "$root/tools/resident-depth/shader-compiler/Cargo.toml"
header_flags=$(pkg-config --cflags vulkan libpng nlohmann_json)
library_flags=$(pkg-config --libs vulkan libpng)
read -ra headers <<< "$header_flags"
read -ra libraries <<< "$library_flags"
read -ra extra_headers <<< "${CXXFLAGS:-}"
read -ra extra_libraries <<< "${LDFLAGS:-}"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -fPIC -shared \
    "${headers[@]}" "${extra_headers[@]}" -I"$CUDA_HOME/include" \
    "$root/tools/resident-depth/vulkan_backend.cpp" "${libraries[@]}" "${extra_libraries[@]}" -lcuda \
    -o "$output/resident-depth/resident_vk.so"
"${ESPALIER_TEXTURE_PYTHON:-python3}" "$root/tools/resident-depth/compile_shaders.py" \
    "$output/resident-depth" "$CARGO_TARGET_DIR/debug/object-shader-compile"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror "${headers[@]}" "${extra_headers[@]}" \
    -isystem "$root/tools/third-party/bcdec" "$root/tools/pack_generated.cpp" \
    "${libraries[@]}" "${extra_libraries[@]}" -o "$output/surfaces/pack-generated"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror "${headers[@]}" "${extra_headers[@]}" \
    "$root/tools/atlas_density.cpp" "${libraries[@]}" "${extra_libraries[@]}" \
    -o "$output/surfaces/atlas-density"

"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror "${headers[@]}" "${extra_headers[@]}" \
    "$root/tools/match_base_colour.cpp" "${libraries[@]}" "${extra_libraries[@]}" \
    -o "$output/surfaces/match-base-colour"
