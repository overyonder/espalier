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
  7. Detail Pass (stages/7_paint.py):    15-view resident projection painting (FLUX.3 or Codex)
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import sys

here = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    p = argparse.ArgumentParser(
        prog="espalier",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("image", nargs="?", help="input image (or views directory for pixal3d)")
    p.add_argument("output", nargs="?", help="output directory or output .glb path")
    p.add_argument("--batch", help="generate base models with each stage's models held across a table of image and work directory")
    p.add_argument("--batch-through", choices=["unwrap", "texture"], default="texture", help="last stage a --batch run performs")
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
    p.add_argument("--paint-base", choices=["trellis", "normals"], default="trellis", help="painting input: textured TRELLIS render (default) or camera-space normals")
    p.add_argument("--base-model", help="paint an existing textured GLB, preserving its mesh and UVs; skip shape and base-texture generation")
    p.add_argument("--base-material", help="use a UV-identical material for painting and colour fallback; retain --base-model geometry, normals and output material")
    p.add_argument("--cameras", help="reuse an exact saved camera rig")
    p.add_argument("--paintings", help="reuse paintings from this directory (requires --cameras)")
    p.add_argument("--painting-output", help="reuse and complete paintings in this directory; requires --cameras")
    p.add_argument("--resume", action="store_true", help="resume completed stages in the output work directory")
    p.add_argument("--work-dir", help="directory for intermediate stage outputs")
    p.add_argument("--upscale", type=int, choices=[0, 2, 4], default=2,
                   help="Real-ESRGAN painting upscale before baking; atlas stays 4096 (default: 2; 0 disables)")
    p.add_argument("--no-base-colour-match", action="store_true", help="disable matching fallback brightness and colour grade to painted views")
    p.add_argument("--texel-density-map", action="store_true",
                   help="write source-pixel density and UV occupancy beside the finished model")

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
    if a.batch:
        if a.model != "hybrid":
            p.error("--batch is supported by the hybrid pipeline")
        return a
    if not a.image or not a.output:
        p.error("image and output are required without --batch")
    if a.gpt_image:
        a.paint = "gpt"

    if a.model == "pixal3d" and a.pipeline not in ("1024_cascade", "1536_cascade"):
        p.error("Pixal3D has only the 1024_cascade and 1536_cascade pipelines")
    if os.path.isdir(a.image) and a.model != "pixal3d":
        p.error("a views directory needs --model pixal3d")
    if a.prompt_file and not os.path.isfile(a.prompt_file):
        p.error(f"Prompt file does not exist: {a.prompt_file}")
    if (a.paintings or a.painting_output) and not a.cameras:
        p.error("--paintings and --painting-output require --cameras so projection matches the paintings")
    if a.base_material and not a.base_model:
        p.error("--base-material requires --base-model")
    if a.paintings and a.painting_output:
        p.error("Choose --paintings or --painting-output")
    if a.base_model and a.model != "hybrid":
        p.error("--base-model is supported by the hybrid pipeline")
    if a.model == "hybrid" and a.texture != 4096:
        p.error("The hybrid baker uses a 4096 atlas; --upscale changes source painting resolution")
    if a.texel_density_map and (a.model != "hybrid" or a.paint == "none"):
        p.error("--texel-density-map requires a painted hybrid run")
    return a


def run_stage(script_name, *stage_args):
    """Executes a pipeline stage script in an isolated subprocess for VRAM reclamation."""
    script_path = os.path.join(here, "stages", script_name)
    role = "SHAPE" if script_name in ("2_generate.py", "3_extract.py") else "TEXTURE"
    python = os.environ.get(f"ESPALIER_{role}_PYTHON", sys.executable)
    environment = os.environ.copy()
    trellis = environment.get("ESPALIER_TRELLIS_ROOT", os.path.join(here, "src", "TRELLIS.2"))
    shape = environment.get("ESPALIER_SHAPE_ROOT", os.path.join(here, "src", "Hunyuan3D-2.1"))
    paths = [trellis, os.path.join(shape, "hy3dshape"), shape, environment.get("PYTHONPATH", "")]
    environment["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    cmd = [python, script_path, *stage_args]
    print(f"START {script_name}", flush=True)
    subprocess.run(cmd, check=True, env=environment, timeout=3600)
    print(f"DONE {script_name}", flush=True)


def generate_batch(a):
    """Runs stages 1-6 over a table, loading each stage's models once. Field volumes are removed once extracted."""
    with open(a.batch) as rows:
        jobs = [row.rstrip("\n").split("\t") for row in rows if row.strip()]
    paths = [(image, *(os.path.join(os.path.abspath(work), name) for name in
                       ("cutout.png", "field.pt", "dense.ply", "reduced.ply", "unwrapped.glb", "trellis.glb")))
             for image, work in jobs]
    def run_batch_stage(script, columns, *arguments):
        with tempfile.NamedTemporaryFile("w", suffix=".tsv") as table:
            table.writelines("\t".join(row) + "\n" for row in columns)
            table.flush()
            run_stage(script, "--batch", table.name, *arguments)
    final = 5 if a.batch_through == "unwrap" else 6
    pending = [p for p in paths if not os.path.isfile(p[final])]
    run_batch_stage("1_cutout.py", [(p[0], p[1]) for p in pending])
    # A finished dense mesh needs no field, which is about half a gigabyte.
    run_batch_stage("2_generate.py", [(p[1], p[2]) for p in pending if not os.path.isfile(p[3])],
                    "--steps", str(a.steps), "--octree", str(a.octree), "--seed", str(a.seed))
    run_batch_stage("3_extract.py", [(p[2], p[3]) for p in pending])
    for p in pending:
        if os.path.isfile(p[3]) and os.path.isfile(p[2]):
            os.remove(p[2])
    run_batch_stage("4_reduce.py", [(p[3], p[4]) for p in pending], "--faces", str(a.faces))
    run_batch_stage("5_unwrap.py", [(p[4], p[5]) for p in pending])
    if final == 6:
        albedo_arguments = ["--texture", str(a.texture)] + (["--dinov3", a.dinov3] if a.dinov3 else [])
        run_batch_stage("6_albedo.py", [(p[1], p[3], p[5], p[6]) for p in pending], *albedo_arguments)
    failed = [p[final] for p in paths if not os.path.isfile(p[final])]
    print(f"BATCH_COMPLETE {len(paths) - len(failed)}/{len(paths)}", *failed, flush=True)


def main():
    a = parse_args()
    if a.batch:
        generate_batch(a)
        return

    if a.model == "hybrid" and a.paint == "flux":
        print(
            "Warning: Local FLUX and Qwen inpainting did not preserve reference "
            "colours adequately in our comparison tests. Our tested workflow "
            "requires API inpainting for good reference-colour fidelity.",
            file=sys.stderr,
            flush=True,
        )

    # Determine output destinations
    if a.output.endswith(".glb"):
        out_dir = os.path.dirname(os.path.abspath(a.output)) or "."
        final_out = os.path.abspath(a.output)
    else:
        out_dir = os.path.abspath(a.output)
        final_out = os.path.join(out_dir, "model.glb")

    os.makedirs(out_dir, exist_ok=True)
    work = os.path.abspath(a.work_dir) if a.work_dir else os.path.join(
        out_dir, os.path.splitext(os.path.basename(final_out))[0] + ".work" if a.output.endswith(".glb") else "work"
    )
    os.makedirs(work, exist_ok=True)

    cutout_png = os.path.join(work, "cutout.png")
    field_pt = os.path.join(work, "field.pt")
    dense_ply = os.path.join(work, "dense.ply")
    reduced_ply = os.path.join(work, "reduced.ply")
    unwrapped_glb = os.path.join(work, "unwrapped.glb")
    base_trellis_glb = os.path.join(work, "trellis.glb")

    if a.model == "hybrid":
        def generate_stage(script, output, *args):
            if a.base_model:
                return
            if a.resume and os.path.isfile(output) and os.path.getsize(output):
                print(f"REUSE {output}", flush=True)
                return
            run_stage(script, *args)

        # Stage 1: Cutout
        cutout_args = [a.image, cutout_png]
        if a.no_rembg:
            cutout_args.append("--no-rembg")
        generate_stage("1_cutout.py", cutout_png, *cutout_args)

        # Stage 2: Shape Generation
        generate_stage("2_generate.py", field_pt, cutout_png, field_pt,
                  "--steps", str(a.steps), "--octree", str(a.octree), "--seed", str(a.seed))

        # Stage 3: Surface Extraction
        generate_stage("3_extract.py", dense_ply, field_pt, dense_ply)

        # Stage 4: Mesh Reduction & Cleaning
        generate_stage("4_reduce.py", reduced_ply, dense_ply, reduced_ply, "--faces", str(a.faces))

        # Stage 5: UV Unwrap
        generate_stage("5_unwrap.py", unwrapped_glb, reduced_ply, unwrapped_glb)

        # Stage 6: Base PBR Texture
        albedo_args = [cutout_png, dense_ply, unwrapped_glb, base_trellis_glb, "--texture", str(a.texture)]
        if a.dinov3:
            albedo_args.extend(["--dinov3", a.dinov3])
        generate_stage("6_albedo.py", base_trellis_glb, *albedo_args)
        if a.base_model:
            base_trellis_glb = os.path.abspath(a.base_model)

        # Stage 7: Detail Painting (if requested)
        if a.paint in ("flux", "gpt"):
            paint_args = [base_trellis_glb, a.image, work, final_out,
                          "--model", a.paint, "--texture", str(a.texture),
                          "--upscale", str(a.upscale)]
            paint_args.extend(["--paint-base", a.paint_base])
            if a.no_base_colour_match:
                paint_args.append("--no-base-colour-match")
            if a.texel_density_map:
                paint_args.append("--texel-density-map")
            for flag, value in (("--base-material", a.base_material), ("--cameras", a.cameras), ("--paintings", a.paintings), ("--painting-output", a.painting_output)):
                if value:
                    paint_args.extend([flag, os.path.abspath(value)])
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
