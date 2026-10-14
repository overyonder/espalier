#!/usr/bin/env bash
# Builds over|yonder Espalier-2: Microsoft's TRELLIS.2 and its CUDA extensions at pinned commits, with Espalier-2's patches applied,
# in a Python 3.10 virtual environment at ./.venv.
#
# Needs: an NVIDIA driver, the CUDA toolkit 12.8 or later (CUDA_HOME, or nvcc on PATH), a GCC that toolkit supports,
# git and uv. TORCH_CUDA_ARCH_LIST defaults to the installed GPU's compute capability.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$here/src
venv=$here/.venv
mkdir -p "$src"

TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
CUMESH_COMMIT=12289e1
FLEXGEMM_COMMIT=6dd94a8
NVDIFFRAST_TAG=v0.4.0
NVDIFFREC_COMMIT=b296927

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)}
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
echo "CUDA_HOME=$CUDA_HOME  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

fetch() {   # fetch <url> <dir> <commit or tag> [patch]
    local url=$1 dir=$src/$2 rev=$3 patch=${4:-}
    if [ ! -d "$dir/.git" ]; then git clone --quiet "$url" "$dir"; fi
    git -C "$dir" checkout --quiet --force "$rev"
    git -C "$dir" submodule update --quiet --init --recursive
    if [ -n "$patch" ]; then git -C "$dir" apply "$here/patches/$patch"; fi
}
fetch https://github.com/microsoft/TRELLIS.2.git TRELLIS.2 "$TRELLIS2_COMMIT" trellis2.patch
fetch https://github.com/JeffreyXiang/CuMesh.git CuMesh "$CUMESH_COMMIT" cumesh.patch
fetch https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM "$FLEXGEMM_COMMIT"
fetch https://github.com/NVlabs/nvdiffrast.git nvdiffrast "$NVDIFFRAST_TAG"
fetch https://github.com/JeffreyXiang/nvdiffrec.git nvdiffrec "$NVDIFFREC_COMMIT"

# PyTorch 2.8 with CUDA 12.8 and flash-attn 2.8.3 are the oldest builds with Blackwell (sm_120) kernels; Microsoft's
# setup pins PyTorch 2.6 with CUDA 12.4 and flash-attn 2.7.3. transformers stays below 5, which this code predates.
[ -d "$venv" ] || uv venv --quiet --python 3.10 "$venv"
uv pip install --python "$venv" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$venv" setuptools wheel ninja imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    trimesh plyfile "transformers==4.57.1" pillow kornia timm zstandard \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
abi=$("$venv/bin/python" -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())")
uv pip install --python "$venv" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.8cxx11abi${abi}-cp310-cp310-linux_x86_64.whl"

# --no-cache --reinstall so the patched sources are always compiled, never a cached build of the originals, and
# --no-deps because o-voxel lists CuMesh and FlexGEMM as unpinned git dependencies that would replace the builds above
for ext in nvdiffrast:nvdiffrast nvdiffrec:nvdiffrec-render CuMesh:cumesh FlexGEMM:flex-gemm TRELLIS.2/o-voxel:o-voxel; do
    dir=${ext%%:*} package=${ext#*:}
    echo "building $dir"
    uv pip install --python "$venv" --no-build-isolation --no-cache --no-deps --reinstall-package "$package" "$src/$dir"
done
site=$("$venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
grep -q CUMESH_REMESH_CHUNK "$site/cumesh/remeshing.py" || { echo "the installed CuMesh is missing Espalier-2's patch"; exit 1; }
grep -q "only needed for baking" "$site/o_voxel/postprocess.py" || { echo "the installed o-voxel is missing Espalier-2's patch"; exit 1; }
echo "Espalier-2 is installed. Run: $here/espalier <image> <output directory>"
