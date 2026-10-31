#!/usr/bin/env python3
"""Stage 3: Extract Dense Surface. Extracts dense manifold mesh from field tensor via GPU Dual Marching Cubes.

  python3 stages/3_extract.py <field.pt> <out_dense.ply>
"""
import argparse, os, sys, time

p = argparse.ArgumentParser()
p.add_argument("field", help="input field tensor (.pt)")
p.add_argument("out", help="output dense mesh (.ply)")
a = p.parse_args()

import torch, trimesh
from skimage import measure

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src", "Hunyuan3D-2.1", "hy3dshape"))
sys.path.insert(0, os.path.join(here, "src", "Hunyuan3D-2.1"))

print("== 3 extract dense surface", flush=True)
kept = torch.load(a.field, map_location="cpu")
k = kept["kwargs"]
started = time.time()

extracted = False
try:
    from hy3dshape.models.autoencoders.surface_extractors import DMCSurfaceExtractor
    extractor = DMCSurfaceExtractor()
    dmc = extractor(kept["grid_logits"].cuda(), **k)[0]
    v, _, _, _ = measure.marching_cubes(kept["grid_logits"][0].numpy(), k["mc_level"], method="lewiner")
    v = v / (k["octree_resolution"] + 1) * (2 * k["bounds"]) - k["bounds"]
    lo, hi = v.min(0), v.max(0)
    dlo, dhi = dmc.mesh_v.min(0), dmc.mesh_v.max(0)
    scale = ((hi - lo) / (dhi - dlo)).mean()
    mesh = trimesh.Trimesh((dmc.mesh_v - (dlo + dhi) / 2) * scale + (lo + hi) / 2, dmc.mesh_f[:, ::-1], process=False)
    extracted = True
except Exception as e:
    print(f"Notice: DMC extraction fallback ({e}); using Lewiner marching cubes", flush=True)

if not extracted:
    v, f, _, _ = measure.marching_cubes(kept["grid_logits"][0].numpy(), k["mc_level"], method="lewiner")
    v = v / (k["octree_resolution"] + 1) * (2 * k["bounds"]) - k["bounds"]
    mesh = trimesh.Trimesh(v, f, process=False)

mesh.export(a.out)
print(f"extracted {len(mesh.vertices)} vertices, {len(mesh.faces)} faces in {time.time() - started:.1f}s -> {a.out}", flush=True)
