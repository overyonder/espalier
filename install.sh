#!/usr/bin/env bash
# Builds over|yonder Espalier-2: Microsoft's TRELLIS.2 and its CUDA extensions at pinned commits, with Espalier-2's patches applied,
# in a Python 3.10 virtual environment at ./.venv. With --pixal3d it also installs Tencent's Pixal3D, which is built on
# TRELLIS.2, into the same environment.
#
# Needs: an NVIDIA driver, the CUDA toolkit 12.8 or later (CUDA_HOME, or nvcc on PATH), a GCC that toolkit supports,
# git and uv, and for --pixal3d also cmake and make. TORCH_CUDA_ARCH_LIST defaults to the installed GPU's compute
# capability.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$here/src
venv=$here/.venv
mkdir -p "$src"
pixal3d=0
for arg in "$@"; do
    case $arg in
        --pixal3d) pixal3d=1 ;;
        *) echo "usage: $0 [--pixal3d]"; exit 1 ;;
    esac
done

TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
CUMESH_COMMIT=12289e1
FLEXGEMM_COMMIT=6dd94a8
NVDIFFRAST_TAG=v0.4.0
NVDIFFREC_COMMIT=b296927
PIXAL3D_COMMIT=f7cf38429b0bd264f1995f0f8743a88b1c728b94
# Pixal3D's requirements take MoGe from its default branch, whose current version conflicts with their trimesh pin; this
# is the commit that was current on the requirements' date, 19 May 2026.
MOGE_COMMIT=07444410f1e33f402353b99d6ccd26bd31e469e8

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
if [ $pixal3d = 1 ]; then
    # Pixal3D's README steps 2 to 4 on the TRELLIS.2 base above. Its gradio requirement is only for its web demo.
    fetch https://github.com/TencentARC/Pixal3D.git Pixal3D "$PIXAL3D_COMMIT" pixal3d.patch
    grep -v -e '^git+https://github.com/microsoft/MoGe' -e '^gradio' "$src/Pixal3D/requirements.txt" > "$src/pixal3d-requirements.txt"
    uv pip install --python "$venv" -r "$src/pixal3d-requirements.txt" "git+https://github.com/microsoft/MoGe.git@$MOGE_COMMIT"
    # MoGe depends on OpenCV's GUI build, which replaces the headless cv2 and needs libGL; reinstalling the headless build
    # puts it back, and MoGe uses nothing the headless build lacks
    uv pip install --python "$venv" --reinstall-package opencv-python-headless \
        "$(grep '^opencv-python-headless' "$src/pixal3d-requirements.txt")"
    CUDACXX=$CUDA_HOME/bin/nvcc CPATH=$CUDA_HOME/include${CPATH:+:$CPATH} NATTEN_CUDA_ARCH=$TORCH_CUDA_ARCH_LIST \
        NATTEN_N_WORKERS=$MAX_JOBS uv pip install --python "$venv" --no-build-isolation natten==0.21.0
    # Pixal3D's own build of utils3d, which MoGe's camera estimation uses; TRELLIS.2's inference doesn't import utils3d
    uv pip install --python "$venv" https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
    for marker in stream_weights_from_file:models/__init__.py TRELLIS_DECODE_TILE_SPLIT:models/sc_vaes/sparse_unet_vae.py \
            NAF_TILE:trainers/flow_matching/mixins/image_conditioned_proj.py sample_index:trainers/flow_matching/mixins/image_conditioned_proj.py \
            "tex_slat = None":pipelines/pixal3d_image_to_3d.py; do
        grep -q "${marker%%:*}" "$src/Pixal3D/pixal3d/${marker#*:}" || { echo "Pixal3D is missing Espalier-2's patch"; exit 1; }
    done
    "$venv/bin/python" -c "import natten, moge, utils3d; utils3d.pt.depth_map_to_point_map" || { echo "Pixal3D's dependencies didn't install"; exit 1; }
fi
echo "Espalier-2 is installed. Run: $here/espalier <image> <output directory>"
