# Source once before a CLI run or batch on NixOS. No model paths or credentials.
espalier_libraries=$(nix build --no-link --print-out-paths nixpkgs#e2fsprogs.out nixpkgs#fontconfig.lib nixpkgs#freetype nixpkgs#gmpxx nixpkgs#libgpg-error nixpkgs#stdenv.cc.cc.lib nixpkgs#libx11 nixpkgs#libxcb nixpkgs#zlib nixpkgs#libglvnd nixpkgs#glib.out nixpkgs#libxkbcommon nixpkgs#p11-kit | sed 's#$#/lib#' | paste -sd:)
espalier_compiler=$(nix build --no-link --print-out-paths nixpkgs#gcc | grep -v -- '-man$')
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$espalier_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="/run/opengl-driver/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
export PATH="$espalier_compiler/bin:$PATH" CC="$espalier_compiler/bin/gcc" CXX="$espalier_compiler/bin/g++"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
unset espalier_libraries espalier_compiler
