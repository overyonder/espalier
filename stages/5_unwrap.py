#!/usr/bin/env python3
"""Stage 5: UV Unwrap. Generates non-overlapping UV atlas layout using xatlas through CuMesh.

  python3 stages/5_unwrap.py <reduced.ply> <out_unwrapped.glb>
"""
import argparse, os, sys, time, torch, numpy as np, trimesh, cumesh
from jobs import run_jobs

p = argparse.ArgumentParser()
p.add_argument("reduced", nargs="?", help="input reduced mesh (.ply)")
p.add_argument("out", nargs="?", help="output unwrapped model (.glb)")
p.add_argument("--batch", help="tab-separated table of the positional paths, holding models across its rows")
a = p.parse_args()

print("== 5 unwrap UVs (xatlas)", flush=True)
def unwrap(reduced_path, out):
    started = time.time()
    m = trimesh.load(reduced_path, process=False, force="mesh")
    mesh = cumesh.CuMesh()
    mesh.init(torch.tensor(np.asarray(m.vertices), dtype=torch.float32).cuda(),
              torch.tensor(np.asarray(m.faces), dtype=torch.int32).cuda())
    vertices, faces, uvs, _ = mesh.uv_unwrap(return_vmaps=True)
    material = trimesh.visual.material.PBRMaterial(baseColorFactor=[115, 110, 105, 255], metallicFactor=0.0, roughnessFactor=0.8)
    trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy(), process=False,
                    visual=trimesh.visual.TextureVisuals(uv=uvs.cpu().numpy(), material=material)).export(out)
    print(f"unwrapped in {time.time() - started:.1f}s -> {out}", flush=True)

run_jobs(a.batch, (a.reduced, a.out), unwrap)
