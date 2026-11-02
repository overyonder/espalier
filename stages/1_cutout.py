#!/usr/bin/env python3
"""Stage 1: Cutout. Removes image background using RMBG-2.0 or validates alpha channel.

  python3 stages/1_cutout.py <image> <out_cutout.png> [--no-rembg]
"""
import argparse, os, sys
from PIL import Image
from jobs import run_jobs

p = argparse.ArgumentParser()
p.add_argument("image", nargs="?", help="input image")
p.add_argument("out", nargs="?", help="output cutout PNG")
p.add_argument("--batch", help="tab-separated table of the positional paths, holding models across its rows")
p.add_argument("--no-rembg", action="store_true", help="skip RMBG-2.0 (input image must have alpha channel)")
a = p.parse_args()

if a.no_rembg:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
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

    def cut_out(image_path, out):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with Image.open(image_path).convert("RGB") as img:
            rm(img).save(out)
        print(f"cutout extracted -> {out}", flush=True)

    run_jobs(a.batch, (a.image, a.out), cut_out)
