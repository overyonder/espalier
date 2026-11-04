#!/usr/bin/env bash
# Builds over|yonder Espalier-2.1: Tencent's Hunyuan3D-2.1 shape generation, Microsoft's TRELLIS.2 PBR texturing and
# progressive projection painting on a 16 GB consumer GPU, with CuMesh and FlexGEMM extensions at pinned commits.
#
# Needs: an NVIDIA driver, the CUDA toolkit 12.8 or later (CUDA_HOME, or nvcc on PATH), a GCC that toolkit supports,
# git and uv. TORCH_CUDA_ARCH_LIST defaults to the installed GPU's compute
# capability.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$here/src
venv=$here/.venv
mkdir -p "$src"
flux=0
for arg in "$@"; do
    case $arg in
        --flux) flux=1 ;;
        *) echo "usage: $0 [--flux]"; exit 1 ;;
    esac
done

TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
CUMESH_COMMIT=12289e1
FLEXGEMM_COMMIT=6dd94a8
NVDIFFRAST_TAG=v0.4.0
NVDIFFREC_COMMIT=b296927
HUNYUAN3D21_COMMIT=82920d643c0dc2f7bfd7255f45f62d386edfe60c
HI3D_COMMIT=13657ad09afaa92e350b461a1cd66399e44a59c4
FLUX_COMMIT=66c403598c1d5300d0246a4e32049e29a39f60fe

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
fetch https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git Hunyuan3D-2.1 "$HUNYUAN3D21_COMMIT"
fetch https://github.com/hi3d-official/Hi3Dv1.0.git Hi3Dv1.0 "$HI3D_COMMIT"

# PyTorch 2.8 with CUDA 12.8 and flash-attn 2.8.3 are the oldest builds with Blackwell (sm_120) kernels; Microsoft's
# setup pins PyTorch 2.6 with CUDA 12.4 and flash-attn 2.7.3. transformers stays below 5, which this code predates.
# Change this environment here: a separate uv pip install doesn't see the transformers pin, and diffusers alone takes
# huggingface-hub 1, which transformers 4 refuses at import.
[ -d "$venv" ] || uv venv --quiet --python 3.10 "$venv"
uv pip install --python "$venv" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$venv" setuptools wheel ninja imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    "numpy<2" "trimesh<5" "opencv-python<4.13" "opencv-python-headless<4.13" pymeshlab "plyfile<1.1" basicsr==1.4.2 realesrgan==0.3.0 "transformers==4.57.1" pillow kornia timm zstandard diffusers accelerate scikit-image einops omegaconf \
    pyfqmr spconv-cu126==2.3.8 \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
abi=$("$venv/bin/python" -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())")
uv pip install --python "$venv" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.8cxx11abi${abi}-cp310-cp310-linux_x86_64.whl"

# --no-cache --reinstall so the patched sources are always compiled, never a cached build of the originals, and
# --no-deps because o-voxel lists CuMesh and FlexGEMM as unpinned git dependencies that would replace the builds above.
# One build at a time, because each already compiles on MAX_JOBS.
extensions=()
for ext in nvdiffrast:nvdiffrast nvdiffrec:nvdiffrec-render CuMesh:cumesh FlexGEMM:flex-gemm TRELLIS.2/o-voxel:o-voxel; do
    extensions+=("$src/${ext%%:*}" --reinstall-package "${ext#*:}")
done
UV_CONCURRENT_BUILDS=1 uv pip install --python "$venv" --no-build-isolation --no-cache --no-deps "${extensions[@]}"

[ -f "$src/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ] || curl -L --create-dirs -o "$src/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
uv pip install --python "$venv" --no-deps -e "$src/Hi3Dv1.0"
# Hi3D refuses a repository holding anything but its own files, and a Hugging Face snapshot also holds .gitattributes.
"$venv/bin/python" - "$src/Hi3Dv1.0/weights" <<'HI3D_WEIGHTS'
import os, sys
from huggingface_hub import hf_hub_download
root = sys.argv[1]
files = ["model_index.json", "scheduler/scheduler_config.json"] + [f"{part}/{name}" for part in ("vae", "transformer_512", "transformer_1024") for name in ("config.json", "model.safetensors")]
for name in files:
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.lexists(path):
        os.symlink(hf_hub_download("hi3d-official/Hi3Dv1.0", name), path)
HI3D_WEIGHTS

# Dual Marching Cubes (diso) for GPU surface extraction
if ! "$venv/bin/python" -c "import diso" 2>/dev/null; then
    echo "building diso"
    uv pip install --python "$venv" --no-build-isolation diso
fi

site=$("$venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
grep -q CUMESH_REMESH_CHUNK "$site/cumesh/remeshing.py" && grep -q "del pts_vert" "$site/cumesh/remeshing.py" \
    && grep -q "edge2face_cnt.free" "$src/CuMesh/src/simplify.cu" || { echo "the installed CuMesh is missing Espalier-2's patch"; exit 1; }
grep -q "only needed for baking" "$site/o_voxel/postprocess.py" && grep -q "the remesh's cached blocks" "$site/o_voxel/postprocess.py" \
    && grep -q "if not bake" "$site/o_voxel/postprocess.py" \
    && grep -q O_VOXEL_MESH_CHUNK "$site/o_voxel/convert/flexible_dual_grid.py" \
    || { echo "the installed o-voxel is missing Espalier-2's patch"; exit 1; }
for marker in stream_weights_from_file:models/__init__.py TRELLIS_DECODE_MLP_CHUNK:models/sc_vaes/sparse_unet_vae.py \
        TRELLIS_DECODE_TILE_SPLIT:models/sc_vaes/sparse_unet_vae.py clear_spatial_caches:pipelines/trellis2_image_to_3d.py; do
    grep -q "${marker%%:*}" "$src/TRELLIS.2/trellis2/${marker#*:}" || { echo "TRELLIS.2 is missing Espalier-2's patch"; exit 1; }
done
if [ $flux = 1 ]; then
    fetch https://github.com/black-forest-labs/flux-3-image-klein.git flux-3-image-klein "$FLUX_COMMIT"
    flux_venv="$src/flux-3-image-klein/.venv"
    [ -d "$flux_venv" ] || uv venv --python 3.12 "$flux_venv"
    uv pip install --python "$flux_venv/bin/python" -e "$src/flux-3-image-klein[nvfp4]" \
        --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://flashinfer.ai/whl/cu130
fi
uv pip check --python "$venv"
"$here/tools/build-native.sh"
echo "Espalier-2.1 is installed. Run: $here/espalier <image> <output directory>"
