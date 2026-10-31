#!/usr/bin/env python3
"""Stage 7: Progressive Projection Painting.

Paints 15 orbit views (8 at 15°, 4 at 50°, 3 at -50°) via FLUX.3 or Codex GPT-image
and projects each view onto the 4096 texture atlas in milliseconds using the native CUDA
projector (project_gpu.so).

Usage:
  python3 stages/7_paint.py <base.glb> <reference.png> <work_dir> <out.glb> [--model flux|gpt] [--texture 4096] [--prompt-file <path>] [--flux-dir <path>]
"""
import argparse, base64, json, os, shutil, subprocess, sys, time, urllib.error, urllib.request
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
p.add_argument("--sharp", type=float, default=4.0, help="winner-take-most exponent (default: 4.0)")
p.add_argument("--prompt-file", default=None, help="path to custom painting prompt")
p.add_argument("--flux-dir", default=None, help="path to flux-3-image-klein directory")
a = p.parse_args()

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prog_dir = os.path.join(a.work, "progressive")
os.makedirs(f"{prog_dir}/gen", exist_ok=True)
os.makedirs(f"{prog_dir}/step", exist_ok=True)

blender_script = os.path.join(here, "tools", "project_paint_blender.py")
align_script = os.path.join(here, "tools", "align_silhouette.sh")

model_label = "Codex GPT-Image" if a.model == "gpt" else "FLUX.3 [klein]"
print(f"== 7 progressive projection painting detail pass (15 views via {model_label})", flush=True)

blender_bin = shutil.which("blender")
if not blender_bin:
    print("Notice: blender not found on PATH; falling back to base texture", file=sys.stderr)
    shutil.copy(a.base, a.out)
    sys.exit(0)

default_prompt_path = os.path.join(here, "prompts", "progressive_paint.txt")
prompt_path = a.prompt_file if a.prompt_file and os.path.exists(a.prompt_file) else default_prompt_path
prompt_text = open(prompt_path, "r", encoding="utf-8").read().strip()

def run_cmd(cmd, tries=3):
    for attempt in range(tries):
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError(f"Command failed: {' '.join(str(x) for x in cmd[:4])}")

blender_command = [blender_bin, "--factory-startup", "-b", "--python-exit-code", "1", "--python", blender_script, "--"]

# 1. Initialize piece.blend
if not os.path.exists(f"{prog_dir}/piece.blend"):
    subprocess.run([*blender_command, "piece", a.base, f"{prog_dir}/piece.blend"], check=True)

# 2. Start resident GPU projector worker
current = f"{prog_dir}/piece.blend"
env_proc = dict(os.environ, LD_LIBRARY_PATH="/run/opengl-driver/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
worker = subprocess.Popen(
    [*blender_command, "serve", current, prog_dir, f"tex={a.texture}", f"res={a.res}", f"sharp={a.sharp}"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    env=env_proc
)

def response():
    for line in worker.stdout:
        if line.startswith("PROJECTOR "):
            return json.loads(line[len("PROJECTOR "):])
        print(line, end="", flush=True)
    raise RuntimeError(f"GPU projector exited unexpectedly with code {worker.wait()}")

def request(val):
    worker.stdin.write(json.dumps(val) + "\n")
    worker.stdin.flush()
    return response()

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
        "prompt": prompt_text,
        "model": "gpt-image-2.5-sunburst",
        "quality": "high",
        "size": "1024x1024",
        "images": [
            {"image_url": f"data:image/png;base64,{base64.b64encode(open(render_path, 'rb').read()).decode('ascii')}"},
            {"image_url": f"data:image/png;base64,{base64.b64encode(open(a.reference, 'rb').read()).decode('ascii')}"},
        ],
    }
    req = urllib.request.Request(
        "https://chatgpt.com/backend-api/codex/images/edits",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
    )
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

started = time.time()
try:
    cams = response()["cams"]
    # 15 fixed cameras: 8 orbit at 15°, 4 at 50°, 3 underside at -50°
    order = (
        [f"a{a:03d}e15" for a in (0, 180, 90, 270, 45, 225, 135, 315)]
        + [f"a{a:03d}e50" for a in (45, 225, 135, 315)]
        + [f"a{a:03d}em50" for a in (0, 120, 240)]
    )
    # Ensure cams present match order
    order = [c for c in order if c in cams]

    for i, name in enumerate(order):
        render_png = f"{prog_dir}/views/{name}.png"
        gen_png = f"{prog_dir}/gen/{name}.png"

        request(dict(op="render", name=name))

        if not os.path.exists(gen_png) or os.path.getsize(gen_png) == 0:
            if a.model == "gpt":
                call_codex_image_edit(render_png, gen_png)
            else:
                call_flux_image_edit(render_png, gen_png)
            # Silhouette alignment
            subprocess.run([align_script, render_png, gen_png, gen_png, "black"], check=True)

        request(dict(op="project", name=name))
        print(f"view {i + 1}/{len(order)} ({name}) painted & projected in {time.time() - started:.0f}s", flush=True)

    request(dict(op="quit"))
    worker.wait(timeout=30)
finally:
    if worker.poll() is None:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()

# Ensure exported GLB from the textured blend
textured_blend = f"{step_dir}/textured.blend"
if not os.path.exists(a.out) or os.path.getsize(a.out) < 1024:
    if os.path.exists(textured_blend):
        subprocess.run([
            "blender", "-b", "--python",
            os.path.join(here, "..", "tools", "project_paint_blender.py"),
            "--", "export", textured_blend, os.path.abspath(a.out)
        ], check=True)

print(f"detail painting complete -> {a.out}", flush=True)
