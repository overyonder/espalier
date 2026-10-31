#!/usr/bin/env python3
"""Stage 7: Progressive Projection Painting. Paints 13 orbit views via FLUX.3 or Codex and projects onto atlas.

  python3 stages/7_paint.py <base.glb> <reference.png> <work_dir> <out.glb> [--model flux|gpt] [--texture 4096] [--prompt-file <path>] [--flux-dir <path>]
"""
import argparse, base64, json, os, shutil, subprocess, sys, time, urllib.error, urllib.request

p = argparse.ArgumentParser()
p.add_argument("base", help="input base GLB")
p.add_argument("reference", help="reference product image")
p.add_argument("work", help="scratch working directory")
p.add_argument("out", help="output final GLB")
p.add_argument("--model", default="flux", choices=["flux", "gpt"], help="image model (default: flux)")
p.add_argument("--texture", type=int, default=4096, help="texture atlas size (default: 4096)")
p.add_argument("--prompt-file", default=None, help="path to custom painting prompt")
p.add_argument("--flux-dir", default=None, help="path to flux-3-image-klein directory")
a = p.parse_args()

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prog_dir = os.path.join(a.work, "progressive")
os.makedirs(f"{prog_dir}/gen", exist_ok=True)
os.makedirs(f"{prog_dir}/step", exist_ok=True)
blender_script = os.path.join(here, "tools", "project_paint_blender.py")

model_label = "Codex GPT-Image" if a.model == "gpt" else "FLUX.3 [klein]"
print(f"== 7 progressive projection painting detail pass (13 views via {model_label})", flush=True)

blender_bin = shutil.which("blender")
if not blender_bin:
    print("Notice: blender not found on PATH; falling back to base texture", file=sys.stderr)
    shutil.copy(a.base, a.out)
    sys.exit(0)

# Resolve painting prompt
default_prompt_path = os.path.join(here, "prompts", "progressive_paint.txt")
prompt_path = a.prompt_file if a.prompt_file and os.path.exists(a.prompt_file) else default_prompt_path
prompt_text = open(prompt_path, "r", encoding="utf-8").read().strip()

def run_b(*cmd_args):
    subprocess.run([blender_bin, "-b", "--python", blender_script, "--", *cmd_args], check=True)

if not os.path.exists(f"{prog_dir}/piece.blend"):
    run_b("piece", a.base, f"{prog_dir}/piece.blend")
if not os.path.exists(f"{prog_dir}/cams.json"):
    run_b("render", f"{prog_dir}/piece.blend", prog_dir, "res=1254")

cams = [c["name"] for c in json.load(open(f"{prog_dir}/cams.json"))["cams"]]
az = [0, 180, 90, 270, 45, 225, 135, 315, 45, 225, 135, 315, 0]
el = [15] * 8 + [50] * 4 + [85]
order = [n for a_val, e_val in zip(az, el) for n in cams if n.startswith(f"a{a_val:03d}e{e_val:02d}")]

current = f"{prog_dir}/piece.blend"
started = time.time()

def call_codex_image_edit(render_path, out_path):
    auth_path = os.environ.get("CODEX_AUTH_PATH", os.path.expanduser("~/.codex/auth.json"))
    with open(auth_path, "r", encoding="utf-8") as f:
        tokens = json.load(f)["tokens"]
    headers = {
        "Authorization": "Bearer " + tokens["access_token"],
        "chatgpt-account-id": tokens["account_id"],
        "originator": "codex_cli_rs",
        "Content-Type": "application/json",
        "User-Agent": "codex_cli_rs/0.159.3",
    }
    body = {
        "prompt": prompt_text, "model": "gpt-image-2.5-sunburst", "quality": "high", "size": "1024x1024",
        "images": [
            {"image_url": f"data:image/png;base64,{base64.b64encode(open(render_path, 'rb').read()).decode('ascii')}"},
            {"image_url": f"data:image/png;base64,{base64.b64encode(open(a.reference, 'rb').read()).decode('ascii')}"},
        ],
    }
    req = urllib.request.Request("https://chatgpt.com/backend-api/codex/images/edits", data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.load(resp)
    open(out_path, "wb").write(base64.b64decode(data["data"][0]["b64_json"]))

def call_flux_image_edit(render_path, out_path):
    candidates = [
        a.flux_dir, os.environ.get("FLUX_DIR"),
        os.path.join(here, "src", "flux-3-image-klein"),
        os.path.expanduser("~/code/models/flux-3-image-klein"),
    ]
    flux = next((os.path.abspath(c) for c in candidates if c and os.path.isdir(c)), None)
    if not flux:
        raise FileNotFoundError("FLUX.3 [klein] checkout not found")
    import tempfile
    with tempfile.TemporaryDirectory() as scratch:
        caption_path = os.path.join(scratch, "caption.txt")
        env = dict(os.environ, PYTHONPATH=f"{os.path.join(flux, 'src')}:{os.environ.get('PYTHONPATH', '')}",
                   WEIGHTS_DIR=os.environ.get("WEIGHTS_DIR", os.path.join(flux, "weights")))
        subprocess.run([sys.executable, os.path.join(flux, "src/flux3/pup.py"),
                        "--prompt", prompt_text, "--image", render_path, "--image", a.reference,
                        "--out", caption_path, "--quantization", "fp8", "--max-model-len", "4096"], check=True, env=env)
        subprocess.run([sys.executable, os.path.join(here, "tools", "flux_paint.py"),
                        caption_path, out_path, "1024", render_path, a.reference], check=True, env=env)

align_script = os.path.join(here, "tools", "align_silhouette.sh")

for i, name in enumerate(order):
    render_png = f"{prog_dir}/views/{name}.png"
    gen_png = f"{prog_dir}/gen/{name}.png"
    run_b("render", current, prog_dir, f"only={name}")

    try:
        if a.model == "gpt":
            call_codex_image_edit(render_png, gen_png)
        else:
            call_flux_image_edit(render_png, gen_png)
    except Exception as e:
        print(f"Notice: painting failed on view {name} ({e}); falling back to base texture", file=sys.stderr)
        shutil.copy(a.base, a.out)
        sys.exit(0)

    subprocess.run([align_script, render_png, gen_png, gen_png, "black"], check=True)
    run_b("project", current, prog_dir, f"tex={a.texture}", f"only={name}", "blend=progressive", "sharp=4")
    shutil.copy(f"{prog_dir}/textured.blend", f"{prog_dir}/step/textured.blend")
    current = f"{prog_dir}/step/textured.blend"
    print(f"view {i + 1}/{len(order)} ({name}) completed in {time.time() - started:.0f}s", flush=True)

run_b("export", current, a.out)
print(f"detail painting complete -> {a.out}", flush=True)
