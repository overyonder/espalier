#!/usr/bin/env python3
"""over|yonder Espalier-2.1: High-fidelity image-to-3D pipelines on a 16 GB consumer GPU.

Master CLI orchestrator directing each stage in an isolated process to ensure complete
GPU memory reclamation between model steps.

  espalier <image> <output directory or out.glb> [--model hybrid|trellis2|pixal3d]
           [--paint flux|gpt|none] [--gpt-image] [--faces 56000] [--texture 4096]
           [--steps 100] [--octree 512] [--seed 12345] [--no-rembg]

Pipeline Stages (under stages/):
  1. Cutout      (stages/1_cutout.py):   RMBG-2.0 background isolation
  2. Shape       (stages/2_generate.py): Hunyuan3D-2.1 flow matching (100 steps, octree 512)
  3. Extract     (stages/3_extract.py):  GPU Dual Marching Cubes dense surface extraction
  4. Reduce      (stages/4_reduce.py):   CuMesh clean and decimation to target faces
  5. Unwrap      (stages/5_unwrap.py):   CuMesh xatlas UV unwrapping
  6. Base PBR    (stages/6_albedo.py):   TRELLIS.2 shape-conditioned PBR texturing (4096 atlas)
  7. Detail Pass (stages/7_paint.py):    13-view progressive projection painting (FLUX.3 or Codex)
"""
import argparse, os, shutil, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    p = argparse.ArgumentParser(
        prog="espalier",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("image", help="input image (or views directory for pixal3d)")
    p.add_argument("output", help="output directory or output .glb path")
    p.add_argument("--model", default="hybrid", choices=["hybrid", "trellis2", "pixal3d"],
                   help="hybrid (Hunyuan3D shape + TRELLIS.2 PBR texture, default), trellis2, or pixal3d")
    p.add_argument("--paint", default="flux", choices=["flux", "gpt", "none"],
                   help="detail pass: 'flux' (FLUX.3 [klein] offline painting, default), 'gpt' (Codex endpoint), or 'none'")
    p.add_argument("--gpt-image", action="store_true", help="shortcut for --paint gpt")
    p.add_argument("--flux-dir", default=None, help="directory holding flux-3-image-klein checkout and weights")
    p.add_argument("--faces", type=int, default=None, help="target face count (default 56000 for hybrid, 1000000 for legacy)")
    p.add_argument("--texture", type=int, default=4096, help="texture atlas resolution (default: 4096)")
    p.add_argument("--seed", type=int, default=None, help="sampling seed (default 12345 for hybrid, 42 for legacy)")
    p.add_argument("--steps", type=int, default=100, help="Hunyuan3D-2.1 inference steps (default: 100)")
    p.add_argument("--octree", type=int, default=512, help="Hunyuan3D-2.1 octree resolution (default: 512)")
    p.add_argument("--no-rembg", action="store_true", help="skip RMBG-2.0; input must already have an alpha channel")
    p.add_argument("--dinov3", default=None, help="alternative DINOv3 Hugging Face repository")
    p.add_argument("--prompt-file", default=None, help="custom progressive repainting prompt file")
    p.add_argument("--upscale", type=int, choices=[0, 2, 4], default=0,
                   help="optional Real-ESRGAN texture atlas upscaling (default: 0, disabled)")

    # Legacy Espalier-2 options:
    p.add_argument("--pipeline", default="1536_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    p.add_argument("--tile", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=49152)
    p.add_argument("--no-texture", action="store_true")
    p.add_argument("--fov", type=float, default=None)

    a = p.parse_args()

    if a.faces is None:
        a.faces = 56000 if a.model == "hybrid" else 1000000
    if a.seed is None:
        a.seed = 12345 if a.model == "hybrid" else 42
    if a.gpt_image:
        a.paint = "gpt"

    if a.model == "pixal3d" and a.pipeline not in ("1024_cascade", "1536_cascade"):
        p.error("Pixal3D has only the 1024_cascade and 1536_cascade pipelines")
    if os.path.isdir(a.image) and a.model != "pixal3d":
        p.error("a views directory needs --model pixal3d")
    return a


def run_stage(script_name, *stage_args):
    """Executes a pipeline stage script in an isolated subprocess for VRAM reclamation."""
    script_path = os.path.join(here, "stages", script_name)
    cmd = [sys.executable, script_path, *stage_args]
    subprocess.run(cmd, check=True)


def main():
    a = parse_args()

    # Determine output destinations
    if a.output.endswith(".glb"):
        out_dir = os.path.dirname(os.path.abspath(a.output)) or "."
        final_out = os.path.abspath(a.output)
    else:
        out_dir = os.path.abspath(a.output)
        final_out = os.path.join(out_dir, "model.glb")

    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "work")
    os.makedirs(work, exist_ok=True)

    cutout_png = os.path.join(work, "cutout.png")
    field_pt = os.path.join(work, "field.pt")
    dense_ply = os.path.join(work, "dense.ply")
    reduced_ply = os.path.join(work, "reduced.ply")
    unwrapped_glb = os.path.join(work, "unwrapped.glb")
    base_trellis_glb = os.path.join(work, "trellis.glb")

    if a.model == "hybrid":
        # Stage 1: Cutout
        cutout_args = [a.image, cutout_png]
        if a.no_rembg:
            cutout_args.append("--no-rembg")
        run_stage("1_cutout.py", *cutout_args)

        # Stage 2: Shape Generation
        run_stage("2_generate.py", cutout_png, field_pt,
                  "--steps", str(a.steps), "--octree", str(a.octree), "--seed", str(a.seed))

        # Stage 3: Surface Extraction
        run_stage("3_extract.py", field_pt, dense_ply)

        # Stage 4: Mesh Reduction & Cleaning
        run_stage("4_reduce.py", dense_ply, reduced_ply, "--faces", str(a.faces))

        # Stage 5: UV Unwrap
        run_stage("5_unwrap.py", reduced_ply, unwrapped_glb)

        # Stage 6: Base PBR Texture
        albedo_args = [cutout_png, dense_ply, unwrapped_glb, base_trellis_glb, "--texture", str(a.texture)]
        if a.dinov3:
            albedo_args.extend(["--dinov3", a.dinov3])
        run_stage("6_albedo.py", *albedo_args)

        # Stage 7: Detail Painting (if requested)
        if a.paint in ("flux", "gpt"):
            paint_args = [base_trellis_glb, a.image, work, final_out,
                          "--model", a.paint, "--texture", str(a.texture),
                          "--upscale", str(a.upscale)]
            if a.prompt_file:
                paint_args.extend(["--prompt-file", a.prompt_file])
            if a.flux_dir:
                paint_args.extend(["--flux-dir", a.flux_dir])
            run_stage("7_paint.py", *paint_args)
        else:
            shutil.copy(base_trellis_glb, final_out)

        print(f"\nEspalier-2.1 complete: {final_out}")

    else:
        # Legacy Espalier-2 models
        for st in ("generate", "export"):
            legacy_args = [a.image, out_dir, "--model", a.model, "--pipeline", a.pipeline,
                           "--seed", str(a.seed), "--texture", str(a.texture),
                           "--faces", str(a.faces), "--stage", st]
            if a.tile is not None:
                legacy_args.extend(["--tile", str(a.tile)])
            if a.no_texture:
                legacy_args.append("--no-texture")
            if a.no_rembg:
                legacy_args.append("--no-rembg")
            if a.dinov3:
                legacy_args.extend(["--dinov3", a.dinov3])
            run_stage("legacy_trellis.py", *legacy_args)


if __name__ == "__main__":
    main()
