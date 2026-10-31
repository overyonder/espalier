#!/usr/bin/env python3
"""Stage 1: Cutout. Removes image background using RMBG-2.0 or validates alpha channel.

  python3 stages/1_cutout.py <image> <out_cutout.png> [--no-rembg]
"""
import argparse, os, sys
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("image", help="input image")
p.add_argument("out", help="output cutout PNG")
p.add_argument("--no-rembg", action="store_true", help="skip RMBG-2.0 (input image must have alpha channel)")
a = p.parse_args()

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

if a.no_rembg:
    img = Image.open(a.image)
    if "A" not in img.getbands():
        sys.exit("Error: --no-rembg was specified but input image has no alpha channel")
    img.save(a.out)
    print(f"cutout preserved -> {a.out}")
else:
    print("== 1 cutout (RMBG-2.0)", flush=True)
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet
    rm = BiRefNet("briaai/RMBG-2.0")
    rm.cuda()
    with Image.open(a.image).convert("RGB") as img:
        res = rm(img)
        res.save(a.out)
    print(f"cutout extracted -> {a.out}")
