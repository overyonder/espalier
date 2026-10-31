# over|yonder Espalier-2.1

over|yonder Espalier-2.1 generates game-ready, high-fidelity 3D assets from a single reference image on a 16 GB consumer
GPU (RTX 5070 Ti).

Earlier image-to-3D models attempted to generate geometry and surface textures in a single latent pass, resulting in
soft, blurry voxel textures and noisy geometry. Espalier-2.1 completely decouples geometry reconstruction from
surface appearance and multi-view projection painting:

1. **Precision Neural Shape**: Tencent's [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) generates a watertight 3D occupancy field at 100 flow matching steps and octree resolution 512.
2. **Dense Surface Extraction**: GPU Dual Marching Cubes (DMC via `diso`) extracts the manifold triangular surface.
3. **Decimation & Repair**: CuMesh cleans duplicate geometry, fills holes, and decimates the dense mesh to a production-ready 56,000 face budget.
4. **UV Unwrapping**: CuMesh and xatlas parameterize and pack clean UV charts into a standard 0–1 texture atlas.
5. **Base PBR Albedo**: Microsoft's [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) is repurposed as a shape-conditioned voxel texturing engine, baking base color, roughness, and metallic into a 4096×4096 atlas without seams.
6. **15-Camera Resident GPU Projection Painting**: 15 multi-angle cameras (8 orbit views at 15°, 4 elevated views at 50°, and 3 underside views at -50°) orbit the model. Each view is rendered, repainted against the reference photograph, silhouette-aligned, and projected onto the 4096 atlas using a resident CUDA C++ kernel in fractions of a second with winner-take-most blending (`w⁴ / (w⁴ + best⁴)`).

Every stage runs in an isolated child process, ensuring that CUDA contexts, model weights, and allocations are completely reclaimed by the operating system before the next stage begins.

![Microsoft's published catapult (top) and Espalier-2.1's output with progressive detail painting (bottom), rendered from the same cameras](docs/catapult-comparison.jpg)

The top row is the GLB published on the Microsoft TRELLIS.2 project page. The bottom row is Espalier-2.1's output from the same input image, showing crisp wood grain, iron hardware, rivets, and torsion rope. Up close, the 4096-pixel textures carry photographic micro-detail:

![The same patch of both catapults from a 3072-pixel render](docs/catapult-closeup.jpg)

## Detail Painting

The detail pass runs multi-view projection painting over the 4096 texture atlas:

- **Offline Local Default (`--paint flux`)**: Uses [FLUX.3 [klein]](https://github.com/black-forest-labs/flux-3-image-klein) with sequential model offloading to run locally on a 16 GB GPU.
- **Codex GPT-Image (`--paint gpt` or `--gpt-image`)**: Uses the ChatGPT plan's Codex endpoint (authenticating through local `~/.codex/auth.json`).
- **Base PBR Only (`--paint none`)**: Skips the detail painting pass and exports the base TRELLIS.2 PBR texture.

The final asset is always exported cleanly to `<output>/model.glb` (or the specified `.glb` path).

## Memory

Measured on an NVIDIA GeForce RTX 5070 Ti (16 GB VRAM) generating the benchmark catapult:

| Stage | Technology | Peak VRAM | Time |
| --- | --- | --- | --- |
| 1. Cutout | RMBG-2.0 | 2.4 GiB | 3s |
| 2. Shape Generation | Hunyuan3D-2.1 (100 steps, octree 512) | 12.8 GiB | 140s |
| 3. Surface Extraction | GPU Dual Marching Cubes (`diso`) | 4.2 GiB | 4s |
| 4. Mesh Reduction | CuMesh cleaning & decimation (56k faces) | 3.8 GiB | 6s |
| 5. UV Unwrapping | CuMesh xatlas UV packing | 2.1 GiB | 5s |
| 6. Base PBR Texture | TRELLIS.2 shape-conditioned voxels (4096²) | 11.9 GiB | 45s |
| 7. Detail Painting | 15-view resident CUDA projector (`project_gpu`) | 5.2 GiB | ~0.03s / proj |

Because each stage runs in an independent child process, peak VRAM never accumulates across models.

## Installing

Espalier-2.1 requires Linux, an NVIDIA GPU with at least 16 GB VRAM, the CUDA toolkit 12.8 or later with a supported GCC, `git`, [`uv`](https://docs.astral.sh/uv/), and [Blender](https://www.blender.org/) on PATH.

```sh
git clone https://github.com/overyonder/espalier
cd espalier
./install.sh
```

`install.sh` clones TRELLIS.2, CuMesh, FlexGEMM, nvdiffrast, nvdiffrec, and Hunyuan3D-2.1 at pinned commits, applies memory-optimization patches, and builds the CUDA extensions into `.venv`.

To install the optional local FLUX.3 [klein] weights and pipeline:
```sh
./install.sh --flux
```

### Hugging Face Access

Two models loaded by the pipeline require accepted terms on Hugging Face:
- [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (or specify `--dinov3 devvcamp07/dinov3-mirror`)
- [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)

Log in once with `hf auth login` before running. If your input image already has an alpha channel, `--no-rembg` skips RMBG-2.0.

## Running

```sh
# Default hybrid pipeline with local FLUX.3 detail painting
./espalier input.png out/

# With Codex GPT-image detail painting
./espalier input.png out/ --paint gpt

# Output directly to a specific GLB file
./espalier input.png path/to/model.glb

# Base geometry and PBR texture only (skip detail painting)
./espalier input.png out/ --paint none
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--paint` | `flux` | Detail painter: `flux` (local FLUX.3 [klein]), `gpt` (Codex endpoint), or `none` |
| `--gpt-image` | off | Shortcut for `--paint gpt` |
| `--faces` | `56000` | Target triangle count for simplified geometry |
| `--texture` | `4096` | Texture atlas resolution (width and height) |
| `--seed` | `12345` | Random seed for shape generation |
| `--steps` | `100` | Hunyuan3D-2.1 flow matching diffusion steps |
| `--octree` | `512` | Hunyuan3D-2.1 occupancy field resolution |
| `--no-rembg` | off | Skip RMBG-2.0 (input image must already have an alpha channel) |
| `--dinov3` | Meta repo | Alternative Hugging Face repo for DINOv3 |
| `--upscale` | `0` | Optional Real-ESRGAN atlas super-resolution (`2` or `4` for 8192 textures; runs in fp16 < 500 MiB VRAM) |
| `--prompt-file` | prompts/progressive_paint.txt | Path to custom detail repainting prompt |
| `--model` | `hybrid` | Pipeline architecture: `hybrid` (Espalier-2.1), `trellis2`, or `pixal3d` |

## Licence

Espalier-2.1 is licensed under MIT-SLOP, MIT with an automated-use condition; see [LICENSE](LICENSE).
TRELLIS.2, its weights, CuMesh, FlexGEMM, Hunyuan3D-2.1, Pixal3D, MoGe, and NATTEN are MIT licensed.
DINOv3 and RMBG-2.0 have their own respective upstream licences.
