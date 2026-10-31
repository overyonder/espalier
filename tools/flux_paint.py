#!/usr/bin/env python3
"""Paints an image with FLUX.3 [klein] using sequential model offloading for a 16 GB GPU.

  python3 tools/flux_paint.py <caption.txt> <out.png> <size> <image...>
"""
import os, sys, time, torch
from pathlib import Path
from PIL import Image

caption, out, size, images = Path(sys.argv[1]).read_text().strip(), sys.argv[2], int(sys.argv[3]), sys.argv[4:]

# Import flux3 pipeline
from flux3.pipeline import MODELS, WEIGHTS, Pipeline, dit_path

shared = WEIGHTS / MODELS["distilled"]["dit"]
started = time.time()
pipeline = Pipeline(
    {"distilled": str(dit_path("distilled", "nvfp4"))},
    str(shared / "text_encoder"),
    str(shared / "ae.safetensors"),
    offload=True,
)
print(f"flux loaded in {time.time() - started:.0f}s", flush=True)

started = time.time()
image = pipeline.generate(
    "distilled",
    caption,
    [Image.open(p).convert("RGB") for p in images],
    size,
    size,
    MODELS["distilled"]["steps"],
    None,
    seed=0,
)
image.save(out)
print(f"painted in {time.time() - started:.0f}s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB -> {out}", flush=True)
