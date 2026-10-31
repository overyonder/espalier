#!/usr/bin/env python3
"""Generates or isolates an object's product image for single-image 3D reconstruction:
one object in a level three-quarter view at low elevation on a solid white background.

  python3 tools/generate_product_image.py <prompt.txt> <product.png> [reference image...] [--size 1024x1024]
"""
import argparse, base64, json, os, sys, time, urllib.error, urllib.request

CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/images/"

p = argparse.ArgumentParser()
p.add_argument("prompt", help="path to prompt file or prompt string")
p.add_argument("out", help="output product image path")
p.add_argument("images", nargs="*", help="optional reference images")
p.add_argument("--size", default="1024x1024", help="image dimensions (default: 1024x1024)")
p.add_argument("--quality", default="high", choices=["high", "medium", "low"])
p.add_argument("--model", default="gpt-image-2.5-sunburst")
a = p.parse_args()

auth_path = os.environ.get("CODEX_AUTH_PATH", os.path.expanduser("~/.codex/auth.json"))
if not os.path.exists(auth_path):
    sys.exit(f"Codex auth file not found at {auth_path}. Please log in via codex.")

with open(auth_path, "r", encoding="utf-8") as f:
    tokens = json.load(f)["tokens"]

prompt_text = open(a.prompt, "r", encoding="utf-8").read().strip() if os.path.isfile(a.prompt) else a.prompt
body = {"prompt": prompt_text, "model": a.model, "quality": a.quality, "size": a.size}

if a.images:
    body["images"] = [
        {
            "image_url": "data:%s;base64,%s" % (
                "image/png" if f.endswith(".png") else "image/jpeg",
                base64.b64encode(open(f, "rb").read()).decode("ascii")
            )
        }
        for f in a.images
    ]

headers = {
    "Authorization": "Bearer " + tokens["access_token"],
    "chatgpt-account-id": tokens["account_id"],
    "originator": "codex_cli_rs",
    "Content-Type": "application/json",
    "User-Agent": "codex_cli_rs/0.159.3",
}

endpoint = CODEX_ENDPOINT + ("edits" if a.images else "generations")
req = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
started = time.time()
try:
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.load(resp)
except urllib.error.HTTPError as e:
    sys.exit(f"Codex request failed with HTTP status {e.code}")
except Exception as e:
    sys.exit(f"Codex request error: {type(e).__name__}")

img_data = base64.b64decode(data["data"][0]["b64_json"])
with open(a.out, "wb") as f:
    f.write(img_data)
print(f"saved product image -> {a.out} ({time.time() - started:.1f}s)")
