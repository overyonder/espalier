# over|yonder Espalier-2

over|yonder Espalier-2 runs Microsoft's [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image-to-3D model at its highest
resolution, the 1536 cascade, on a 16 GB consumer GPU. Microsoft lists a 24 GB GPU as the minimum, and on a 16 GB card
its 1536 pipeline runs out of memory. With Espalier-2's patches the same pipeline, with the same weights and settings,
generates and exports a textured GLB on an RTX 5070 Ti.

An espalier is a fruit tree trained flat against a trellis, so a full-size tree grows in a narrow space.

![Microsoft's published catapult (top) and Espalier-2's at 1536 on a 16 GB card (bottom), rendered from the same cameras](docs/catapult-comparison.jpg)

The top row is the GLB Microsoft publishes for this input on the TRELLIS.2 project page. The bottom row is Espalier-2's
output from the same input image at 1536, with seed 42. Microsoft doesn't publish the seed or resolution of its
gallery items, so the two are different samples of the same model. Up close, the 4096-pixel textures carry the same
level of detail:

![The same patch of both catapults from a 3072-pixel render](docs/catapult-closeup.jpg)

The article [over|yonder Espalier-2: TRELLIS.2 at 1536 on a 16 GB GPU](https://over-yonder.tech/games/espalier/) explains how it
works, step by step, with every measurement.

## Memory

Measured on an RTX 5070 Ti (16 GB) generating the catapult above, with the desktop using another 2.1 GiB of the card.
The process peaks come from the NVIDIA driver, sampled four times a second, and include CUDA's own allocations.

| Stage | Microsoft's code | Espalier-2 |
| --- | --- | --- |
| Generation (flow models and decoders) | out of memory in the shape decoder | 11.9 GiB process peak |
| GLB export (remesh, simplify, bake) | out of memory in the remesh | 13.3 GiB process peak |

Generation took about 5 minutes and the export about 1 minute. Microsoft gives about 60 seconds on an H100.

## Changes

Espalier-2 changes three places in Microsoft's code and CuMesh, and runs generation and export as separate processes.
The weights, samplers, seeds and export settings are Microsoft's.

| Where | Change | Effect on the output |
| --- | --- | --- |
| Shape and texture decoders (`SparseUnetVaeDecoder`) | The two highest-resolution stages run in overlapping spatial tiles. | Floating-point differences: at 1024 the largest feature difference is 0.016 on a 0 to 1 scale, and 0.002% of voxels differ. A single tile reproduces the untiled output exactly. |
| Narrow-band remesh (`cumesh.remeshing.remesh_narrow_band_dc`) | Its connectivity and triangle steps run over chunks of voxels and quads. | None. The output is bit-identical. |
| GLB export (`o_voxel.postprocess.to_glb`) | The texture volume stays in system memory until the bake, and the previous mesh is released before the remeshed one is built. | None. |
| Running the pipeline (`espalier.py`) | Generation and export run in separate processes. | None. |

The patches are in [patches/](patches), against TRELLIS.2 `75fbf01` and CuMesh `12289e1`.

## Installing

Espalier-2 needs Linux, an NVIDIA GPU with 16 GB, the CUDA toolkit 12.8 or later with a GCC it supports, `git` and
[`uv`](https://docs.astral.sh/uv/). It has been run on an RTX 5070 Ti (Blackwell, compute capability 12.0).

```sh
git clone https://github.com/overyonder/espalier
cd espalier
./install.sh
```

`install.sh` clones TRELLIS.2, CuMesh, FlexGEMM, nvdiffrast and nvdiffrec at fixed commits, applies the patches, and
builds the CUDA extensions into `.venv` for your GPU's compute capability. It installs PyTorch 2.8 with CUDA 12.8 and
flash-attn 2.8.3, the first releases with kernels for Blackwell GPUs.

Two of the models TRELLIS.2 loads are gated on Hugging Face. Request access to
[DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and
[RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0), then log in with `hf auth login`. RMBG-2.0 only removes the
background of images without an alpha channel, and its licence is non-commercial. For images that already have
transparency, `--no-rembg` skips it.

## Running

```sh
./espalier input.png out/
```

This writes `out/model.glb`, plus `out/raw.pt` with the full-resolution mesh and voxel attributes. The options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--pipeline` | `1536_cascade` | Any of Microsoft's pipelines: `512`, `1024`, `1024_cascade`, `1536_cascade` |
| `--seed` | 42 | Sampling seed |
| `--tile` | 256 at 1536, off otherwise | Decoder tile edge in voxels; 0 turns tiling off |
| `--texture` | 4096 | Texture size in the GLB |
| `--faces` | 1000000 | Face target of the GLB |
| `--no-rembg` | off | Don't load RMBG-2.0; the input needs an alpha channel |
| `--dinov3` | Meta's repository | Another Hugging Face repository holding DINOv3 ViT-L/16 |

## Licence

Espalier-2 is MIT licensed, as are TRELLIS.2, its weights, CuMesh and FlexGEMM. DINOv3 and RMBG-2.0 have their own
licences.
