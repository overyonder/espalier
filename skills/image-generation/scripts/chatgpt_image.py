"""Generate or edit an image through the ChatGPT plan's Codex images endpoint, using the Codex login in ~/.codex/auth.json.

  chatgpt_image.py PROMPT_FILE OUT.png [IMAGE ...] [--size WxH] [--quality high|medium|low] [--model NAME]

With input images it is an edit (images in order: subject, then references); without, a generation. The endpoint
accepts the model name but does not report which model ran, and may serve a lower quality than requested.
"""
import argparse, base64, json, os, sys, time, urllib.error, urllib.request

ENDPOINT = "https://chatgpt.com/backend-api/codex/images/"

p = argparse.ArgumentParser()
p.add_argument("prompt"); p.add_argument("out"); p.add_argument("images", nargs="*")
p.add_argument("--size", default="2048x1152"); p.add_argument("--quality", default="high"); p.add_argument("--model", default="gpt-image-2.5-sunburst")
a = p.parse_args()

tokens = json.load(open(os.path.expanduser("~/.codex/auth.json")))["tokens"]
body = {"prompt": open(a.prompt).read(), "model": a.model, "quality": a.quality, "size": a.size}
if a.images:
    body["images"] = [{"image_url": "data:%s;base64,%s" % ("image/png" if f.endswith(".png") else "image/jpeg",
                                                          base64.b64encode(open(f, "rb").read()).decode())} for f in a.images]
req = urllib.request.Request(ENDPOINT + ("edits" if a.images else "generations"), data=json.dumps(body).encode(), headers={
    "Authorization": "Bearer " + tokens["access_token"], "chatgpt-account-id": tokens["account_id"],
    "originator": "codex_cli_rs", "Content-Type": "application/json", "User-Agent": "codex_cli_rs/0.159.3"})
start = time.time()
try:
    d = json.load(urllib.request.urlopen(req, timeout=900))
except urllib.error.HTTPError as e:
    sys.exit("HTTP %d %s" % (e.code, e.read()[:600].decode(errors="replace")))
open(a.out, "wb").write(base64.b64decode(d["data"][0]["b64_json"]))
print(a.out, "%ds" % (time.time() - start), d.get("quality"), d.get("size"))
