#!/usr/bin/env python3
"""Stage 6: UV Unwrap. Generates non-overlapping UV atlas layout using xatlas through CuMesh.

  python3 stages/6_unwrap.py <reduced.ply> <out_unwrapped.glb>
"""
import argparse, os, sys, time, torch, numpy as np, trimesh, cumesh
from jobs import run_jobs

p = argparse.ArgumentParser()
p.add_argument("reduced", nargs="?", help="input reduced mesh (.ply)")
p.add_argument("out", nargs="?", help="output unwrapped model (.glb)")
p.add_argument("--batch", help="tab-separated table of the positional paths, holding models across its rows")
a = p.parse_args()

print("== 6 unwrap UVs (xatlas)", flush=True)
def unwrap(reduced_path, out):
    started = time.time()
    m = trimesh.load(reduced_path, process=False, force="mesh")
    mesh = cumesh.CuMesh()
    mesh.init(torch.tensor(np.asarray(m.vertices), dtype=torch.float32).cuda(),
              torch.tensor(np.asarray(m.faces), dtype=torch.int32).cuda())
    vertices, faces, uvs, _ = mesh.uv_unwrap(return_vmaps=True)
    material = trimesh.visual.material.PBRMaterial(baseColorFactor=[115, 110, 105, 255], metallicFactor=0.0, roughnessFactor=0.8)
    unwrapped = trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy(), process=False,
                                visual=trimesh.visual.TextureVisuals(uv=uvs.cpu().numpy(), material=material))
    # Vertices split along an atlas seam share one normal, or the seam shows as a crease.
    _, welded = np.unique(unwrapped.vertices, axis=0, return_inverse=True)
    normals = np.zeros((welded.max() + 1, 3))
    np.add.at(normals, welded, unwrapped.vertex_normals)
    unwrapped.vertex_normals = (normals / np.linalg.norm(normals, axis=1, keepdims=True))[welded]
    unwrapped.export(out, include_normals=True)
    print(f"unwrapped in {time.time() - started:.1f}s -> {out}", flush=True)

run_jobs(a.batch, (a.reduced, a.out), unwrap)
