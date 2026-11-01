#!/usr/bin/env python3
"""Generates or isolates an object's product image for single-image 3D reconstruction:
one object in a level three-quarter view at low elevation on a solid white background.

  python3 tools/generate_product_image.py <prompt.txt> <product.png> [reference image...] [--size 1024x1024]
"""
import argparse
from pathlib import Path
from chatgpt_image import CodexImageClient

p = argparse.ArgumentParser()
p.add_argument("prompt", help="path to prompt file or prompt string")
p.add_argument("out", help="output product image path")
p.add_argument("images", nargs="*", help="optional reference images")
p.add_argument("--size", default="1024x1024", help="image dimensions (default: 1024x1024)")
p.add_argument("--quality", default="high", choices=["high", "medium", "low"])
p.add_argument("--model", default="gpt-image-2.5-sunburst")
a = p.parse_args()

prompt_text = Path(a.prompt).read_text().strip() if Path(a.prompt).is_file() else a.prompt
CodexImageClient().generate(prompt_text, a.out, a.images, a.size, a.quality, a.model)
