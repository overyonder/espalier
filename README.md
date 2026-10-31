# over|yonder Espalier-2.1

over|yonder Espalier-2.1 generates game-ready, high-fidelity 3D assets on a 16 GB consumer GPU. It combines Tencent's
[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) for high-accuracy shape generation, Microsoft's
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) for base PBR voxel texturing, and an optional 13-view progressive
projection painting pass for sharp, photographic surface detail.

An espalier is a fruit tree trained flat against a trellis, so a full-size tree grows in a narrow space.

## The Hybrid Pipeline

Espalier-2.1 uses a hybrid single-image reconstruction and texturing pipeline:

1. **Cutout**: [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) extracts the object silhouette from the reference image.
2. **Shape Generation**: Tencent's Hunyuan3D-2.1 flow matching at 100 inference steps and octree resolution 512 decodes the high-resolution occupancy field.
3. **Dense Surface Extraction**: GPU Dual Marching Cubes (DMC via `diso`) extracts the manifold surface and aligns its bounding box to the coordinate frame.
4. **Mesh Cleaning and Decimation**: CuMesh cleans duplicate faces and non-manifold edges, fills holes, and simplifies the dense mesh to 56,000 target triangles.
5. **UV Unwrapping**: CuMesh unwraps the simplified mesh with xatlas into a single UV atlas.
6. **Base PBR Texture**: TRELLIS.2's shape-conditioned texturing pipeline samples PBR voxels (base colour, metallic, and roughness) from the dense geometry directly into a 4096×4096 texture atlas with edge dilation inpainting.
7. **Detail Pass (Progressive Painting)**: 13 multi-angle Blender camera renders orbit the model (8 at 15° elevation, 4 at 50°, and 1 overhead). Each view is rendered, repainted against the reference photo, silhouette-aligned, and projected onto the 4096 atlas with winner-take-most blending (`w⁴ / (w⁴ + best⁴)`).

The stages run in isolated processes so CUDA allocations and PyTorch cache from one stage are completely released by the operating system before the next stage begins.

## Detail Painting

The detail pass runs progressive projection painting over the 4096 texture atlas:

- **Offline Default (`--paint flux`)**: Uses [FLUX.3 [klein]](https://github.com/black-forest-labs/flux-3-image-klein) with prompt upsampling and sequential model offloading to run locally on a 16 GB GPU.
- **Codex GPT-Image (`--paint gpt` or `--gpt-image`)**: Uses the ChatGPT plan's Codex image editing CLI endpoint (authenticating through local `~/.codex/auth.json`).
- **Base Texture Only (`--paint none`)**: Skips projection painting and exports the base TRELLIS.2 PBR texture.

The final asset is always exported cleanly to `<output>/model.glb` (or the specified `.glb` path). API credentials and session tokens are read only from local user storage and are never printed to terminal logs, error traces, or written to disk.

## Installing

Espalier-2.1 requires Linux, an NVIDIA GPU with at least 16 GB VRAM, the CUDA toolkit 12.8 or later with a supported GCC, `git`, [`uv`](https://docs.astral.sh/uv/), and [Blender](https://www.blender.org/) on PATH for progressive painting. It has been verified on an RTX 5070 Ti (Blackwell, compute capability 12.0).

```sh
git clone https://github.com/overyonder/espalier
cd espalier
./install.sh
```

`install.sh` clones TRELLIS.2, CuMesh, FlexGEMM, nvdiffrast, nvdiffrec, and Hunyuan3D-2.1 at pinned commits, applies memory-optimization patches to TRELLIS.2 and CuMesh, and builds the CUDA extensions into `.venv`.

### Hugging Face Access

Two models loaded by the pipeline require accepted terms on Hugging Face:
- [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (or specify `--dinov3 devvcamp07/dinov3-mirror`)
- [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)

Log in once with `hf auth login` before running. If your input image already has transparency, `--no-rembg` skips RMBG-2.0 entirely.

## Running

```sh
# Run with default offline pipeline (FLUX.3 [klein] progressive detail painting)
./espalier input.png out/

# Run with GPT-image detail painting (ChatGPT plan Codex endpoint)
./espalier input.png out/ --paint gpt

# Run without projection painting (base TRELLIS.2 PBR texture only)
./espalier input.png out/ --paint none
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `hybrid` | `hybrid` (Hunyuan3D-2.1 shape + TRELLIS.2 texture), or legacy `trellis2` / `pixal3d` |
| `--paint` | `flux` | Detail pass: `flux` (FLUX.3 [klein] offline painting, default), `gpt` (Codex endpoint), or `none` |
| `--gpt-image` | off | Shortcut flag for `--paint gpt` |
| `--flux-dir` | detected | Path to local `flux-3-image-klein` checkout |
| `--faces` | 56000 | Target face count for mesh decimation (default 56,000 for hybrid, 1,000,000 for legacy models) |
| `--texture` | 4096 | Texture atlas resolution of the exported GLB |
| `--steps` | 100 | Hunyuan3D-2.1 inference steps for shape generation |
| `--octree` | 512 | Hunyuan3D-2.1 octree resolution |
| `--seed` | 12345 | Sampling seed (12345 for hybrid, 42 for legacy) |
| `--no-rembg` | off | Skip RMBG-2.0 background removal (input image must have an alpha channel) |
| `--dinov3` | Meta repository | Hugging Face repository to load DINOv3 ViT-L/16 from |
| `--prompt-file` | `prompts/progressive_paint.txt` | Path to custom progressive repainting prompt |

## Prompts, Tools & Skills

The repository bundles the prompts, tooling, and skills used during asset production:

- **Prompts (`prompts/`)**:
  - `progressive_paint.txt`: The production prompt for 13-view progressive projection painting over rendered base textures.
  - `turnaround_paint.txt`: Four-view turnaround sheet repainting prompt.
  - `product_image_generation.txt`: Standard studio prompt for generating single-object product shots on pure white backgrounds.
  - `product_image_extraction.txt`: Extraction prompt for isolating an assembly from environment or scene reference artwork.
- **Skills (`skills/`, `.claude/skills/`)**:
  - `image-generation`: The agent skill and client script for running image generation and edits through the ChatGPT plan Codex endpoint (`skills/image-generation/scripts/chatgpt_image.py`).
- **Tools (`tools/`)**:
  - `project_paint_blender.py`: Headless Blender script driving orbit camera rendering, UV position/normal baking, and progressive projection blending.
  - `align_silhouette.sh`: Silhouette locator and alignment tool ensuring projected views match render silhouettes pixel-for-pixel.
  - `generate_product_image.py`: Command-line tool to create or isolate clean three-quarter product images from prompts and references.

### Legacy Pipelines

Espalier-2's standalone pipelines remain supported:

```sh
# Pure TRELLIS.2 1536 cascade on 16 GB GPU
./espalier input.png out/ --model trellis2

# Tencent Pixal3D
./espalier input.png out/ --model pixal3d
```

## Licence

Espalier-2.1 is licensed under MIT-SLOP, MIT with an automated-use condition; see [LICENSE](LICENSE). Upstream components retain their respective licences: TRELLIS.2, CuMesh, FlexGEMM, and Pixal3D are MIT licensed; Hunyuan3D-2.1 is licensed under the Tencent Hunyuan3D community licence; DINOv3 and RMBG-2.0 have their own terms.
