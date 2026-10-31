#!/usr/bin/env python3
"""Stage 4: Reduce & Clean Mesh. Cleans manifold topology and simplifies geometry to target face count.

  python3 stages/4_reduce.py <dense.ply> <out_reduced.ply> [--faces 56000]
"""
import argparse, os, sys, time, torch, numpy as np, trimesh, cumesh

p = argparse.ArgumentParser()
p.add_argument("dense", help="input dense mesh (.ply)")
p.add_argument("out", help="output reduced mesh (.ply)")
p.add_argument("--faces", type=int, default=56000, help="target face count (default: 56000)")
a = p.parse_args()

print(f"== 4 reduce and clean (target: {a.faces} faces)", flush=True)

def clean(m):
    m.remove_duplicate_faces()
    m.repair_non_manifold_edges()
    m.remove_small_connected_components(1e-5)
    m.fill_holes(max_hole_perimeter=3e-2)

started = time.time()
m = trimesh.load(a.dense, process=False)
vertices = torch.tensor(np.asarray(m.vertices), dtype=torch.float32).cuda()
faces = torch.tensor(np.asarray(m.faces), dtype=torch.int32).cuda()
del m

mesh = cumesh.CuMesh()
mesh.init(vertices, faces)
mesh.fill_holes(max_hole_perimeter=3e-2)
mesh.simplify(a.faces * 3)
clean(mesh)
mesh.simplify(a.faces)
clean(mesh)
mesh.unify_face_orientations()
v, f = mesh.read()

reduced = trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy(), process=False)
reduced.export(a.out)
print(f"reduced to {len(v)} vertices, {len(f)} faces in {time.time() - started:.1f}s -> {a.out}", flush=True)
