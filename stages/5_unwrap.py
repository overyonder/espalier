#!/usr/bin/env python3
"""Stage 5: UV Unwrap. Generates non-overlapping UV atlas layout using xatlas through CuMesh.

  python3 stages/5_unwrap.py <reduced.ply> <out_unwrapped.glb>
"""
import argparse, os, sys, time, torch, numpy as np, trimesh, cumesh

p = argparse.ArgumentParser()
p.add_argument("reduced", help="input reduced mesh (.ply)")
p.add_argument("out", help="output unwrapped model (.glb)")
a = p.parse_args()

print("== 5 unwrap UVs (xatlas)", flush=True)
started = time.time()
m = trimesh.load(a.reduced, process=False, force="mesh")
mesh = cumesh.CuMesh()
mesh.init(torch.tensor(np.asarray(m.vertices), dtype=torch.float32).cuda(),
          torch.tensor(np.asarray(m.faces), dtype=torch.int32).cuda())
vertices, faces, uvs, _ = mesh.uv_unwrap(return_vmaps=True)
material = trimesh.visual.material.PBRMaterial(baseColorFactor=[115, 110, 105, 255], metallicFactor=0.0, roughnessFactor=0.8)
trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy(), process=False,
                visual=trimesh.visual.TextureVisuals(uv=uvs.cpu().numpy(), material=material)).export(a.out)
print(f"unwrapped in {time.time() - started:.1f}s -> {a.out}", flush=True)
