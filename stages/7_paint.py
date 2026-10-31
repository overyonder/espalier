#!/usr/bin/env python3
"""Stage 7: Detail Painting & GPU Projection Baking.

Renders 15 multi-angle views of the textured base, refines each view with FLUX.3 or Codex GPT-Image,
and bakes all 15 views simultaneously onto the 4096 atlas using the resident Vulkan GPU projection baker.

Usage:
  python3 stages/7_paint.py <base.glb> <reference.png> <work_dir> <out.glb> [--model flux|gpt] [--texture 4096] [--prompt-file <path>] [--flux-dir <path>]
"""
import argparse, base64, json, os, shutil, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import Future
from pathlib import Path

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"

p = argparse.ArgumentParser()
p.add_argument("base", help="input base GLB")
p.add_argument("reference", help="reference product image")
p.add_argument("work", help="scratch working directory")
p.add_argument("out", help="output final GLB")
p.add_argument("--model", default="flux", choices=["flux", "gpt"], help="image model (default: flux)")
p.add_argument("--texture", type=int, default=4096, help="texture atlas size (default: 4096)")
p.add_argument("--res", type=int, default=1254, help="render view resolution (default: 1254)")
p.add_argument("--upscale", type=int, choices=[0, 2, 4], default=0, help="upscale texture atlas with Real-ESRGAN (default: 0, disabled)")
p.add_argument("--prompt-file", default=None, help="path to custom painting prompt")
p.add_argument("--flux-dir", default=None, help="path to flux-3-image-klein directory")
a = p.parse_args()

here = Path(__file__).resolve().parent.parent
tools_dir = here / "tools"
rd_dir = tools_dir / "resident-depth"
sys.path.insert(0, str(rd_dir))

prog_dir = Path(a.work) / "progressive"
views_dir = prog_dir / "views"
gen_dir = prog_dir / "gen"
bake_dir = prog_dir / "bake"
views_dir.mkdir(parents=True, exist_ok=True)
gen_dir.mkdir(parents=True, exist_ok=True)
bake_dir.mkdir(parents=True, exist_ok=True)

model_label = "Codex GPT-Image" if a.model == "gpt" else "FLUX.3 [klein]"
print(f"== 7 15-view detail painting & resident GPU projection bake via {model_label}", flush=True)

# 1. Render 15 camera views if not already rendered
cams_file = prog_dir / "cams.json"
if not cams_file.is_file():
    from batch import fit_cameras, load_geometry
    vertices, _ = load_geometry(Path(a.base), False)
    camera_data = fit_cameras(vertices)
    cams_file.write_text(json.dumps(camera_data, indent=2))
else:
    camera_data = json.loads(cams_file.read_text())

cams = camera_data["cams"]

blender_bin = shutil.which("blender")
if blender_bin and not all((views_dir / f"{c['name']}.png").is_file() for c in cams):
    subprocess.run([
        blender_bin, "--factory-startup", "-b", "--python-exit-code", "1",
        "--python", str(tools_dir / "project_paint_blender.py"),
        "--", "render", a.base, str(prog_dir), f"res={a.res}"
    ], check=True)

default_prompt_path = here / "prompts" / "progressive_paint.txt"
prompt_path = Path(a.prompt_file) if a.prompt_file and Path(a.prompt_file).is_file() else default_prompt_path
prompt_text = prompt_path.read_text(encoding="utf-8").strip()

def call_codex_image_edit(render_path, out_path):
    token = None
    for p_path in [Path.home() / ".codex/auth.json", Path.home() / ".config/codex/auth.json"]:
        if p_path.is_file():
            token = json.loads(p_path.read_text()).get("tokens", {}).get("access_token")
            if token: break
    if not token:
        raise RuntimeError("Codex auth token not found; please log in with codex")

    with open(render_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(a.reference, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode("utf-8")

    body = json.dumps({
        "model": "gpt-image",
        "prompt": prompt_text,
        "images": [
            {"data": img_b64, "mime_type": "image/png"},
            {"data": ref_b64, "mime_type": "image/png"}
        ],
        "size": "1024x1024",
        "output_format": "png"
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/images/edits",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        out_b64 = data["data"][0]["b64_json"]
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(out_b64))

def call_flux_image_edit(render_path, out_path):
    flux_script = tools_dir / "flux_paint.py"
    cmd = [sys.executable, str(flux_script), render_path, a.reference, out_path, "--prompt", prompt_text]
    if a.flux_dir:
        cmd.extend(["--flux-dir", a.flux_dir])
    subprocess.run(cmd, check=True)

# 2. Refine all 15 views
align_script = tools_dir / "align_silhouette.sh"
for i, c in enumerate(cams):
    name = c["name"]
    render_png = str(views_dir / f"{name}.png")
    gen_png = str(gen_dir / f"{name}.png")
    if not os.path.exists(gen_png) or os.path.getsize(gen_png) == 0:
        if a.model == "gpt":
            call_codex_image_edit(render_png, gen_png)
        else:
            call_flux_image_edit(render_png, gen_png)
        if align_script.is_file():
            subprocess.run([str(align_script), render_png, gen_png, gen_png, "black"], check=True)
        print(f"view {i+1}/{len(cams)} ({name}) painted", flush=True)

# 3. Resident Vulkan GPU projection bake
print("Baking 15 views with resident Vulkan GPU projection...", flush=True)
import nvdiffrast.torch as dr
import torch
from batch import load_geometry
from resident import camera_matrices
from resident_vk import bake

geom_path = Path(a.base)
vertices, normals, uv, faces, flat = load_geometry(geom_path, True)
matrices = camera_matrices(camera_data, "cuda")
context = dr.RasterizeCudaContext()

paint_paths = [gen_dir / f"{c['name']}.png" for c in cams]
future = Future()
future.set_result((vertices, normals, uv, faces))

bake(future, matrices, camera_data, paint_paths, bake_dir, context, flat_normals=flat)
print("GPU projection bake complete", flush=True)

# 4. Pack output GLB
pack_bin = tools_dir / "Intermediate/surfaces/pack-generated"
if not pack_bin.is_file():
    pack_bin = tools_dir / "pack-generated"

out_path = Path(a.out).resolve()
out_path.parent.mkdir(parents=True, exist_ok=True)

if pack_bin.is_file():
    subprocess.run([
        str(pack_bin),
        str(bake_dir / "sparse-uv.ply"),
        str(bake_dir / "atlas-bc7.ktx"),
        str(bake_dir / "bake-timings.json"),
        str(out_path)
    ], check=True)
else:
    shutil.copy(a.base, out_path)

if a.upscale > 0:
    print(f"Upscaling texture atlas with Real-ESRGAN (factor={a.upscale})...", flush=True)
    sys.path.insert(0, str(tools_dir))
    from upscale import load_model, upscale_glb
    model = load_model()
    upscale_glb(out_path, out_path, model, factor=a.upscale)

print(f"detail painting complete -> {out_path}", flush=True)
