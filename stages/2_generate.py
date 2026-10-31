#!/usr/bin/env python3
"""Stage 2: Shape Generation. Generates 3D occupancy field using Hunyuan3D-2.1 flow matching.

  python3 stages/2_generate.py <cutout.png> <out_field.pt> [--steps 100] [--octree 512] [--seed 12345]
"""
import argparse, os, sys, time, torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("cutout", help="input cutout PNG")
p.add_argument("out", help="output field tensor (.pt)")
p.add_argument("--steps", type=int, default=100, help="inference steps (default: 100)")
p.add_argument("--octree", type=int, default=512, help="octree resolution (default: 512)")
p.add_argument("--seed", type=int, default=12345, help="sampling seed (default: 12345)")
a = p.parse_args()

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src", "Hunyuan3D-2.1", "hy3dshape"))
sys.path.insert(0, os.path.join(here, "src", "Hunyuan3D-2.1"))

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

print(f"== 2 generate shape (Hunyuan3D-2.1, {a.steps} steps, octree {a.octree}, seed {a.seed})", flush=True)
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")

class FieldSaved(Exception):
    pass

def save_field(latents, **kwargs):
    grid_logits = pipe.vae.volume_decoder(latents, pipe.vae.geo_decoder, **kwargs)
    torch.save({"latents": latents.cpu(), "grid_logits": grid_logits.cpu(), "kwargs": kwargs}, a.out)
    raise FieldSaved(tuple(grid_logits.shape))

pipe.vae.latents2mesh = save_field

started = time.time()
try:
    with Image.open(a.cutout) as image:
        pipe(image=image.convert("RGBA"), num_inference_steps=a.steps, octree_resolution=a.octree,
             generator=torch.manual_seed(a.seed), output_type="trimesh")
except FieldSaved as field:
    print(f"saved field {field.args[0]} in {time.time() - started:.1f}s -> {a.out}", flush=True)
