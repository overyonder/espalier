#!/usr/bin/env python3
"""Stage 5: Reduce & Clean Mesh. Cleans manifold topology and simplifies geometry to target face count.

  python3 stages/5_reduce.py <refined.ply> <out_reduced.ply> [--faces 56000]
"""
import argparse, os, sys, time, torch, numpy as np, trimesh, cumesh
from jobs import run_jobs

p = argparse.ArgumentParser()
p.add_argument("dense", nargs="?", help="input refined or dense mesh (.ply)")
p.add_argument("out", nargs="?", help="output reduced mesh (.ply)")
p.add_argument("--batch", help="tab-separated table of the positional paths, holding models across its rows")
p.add_argument("--faces", type=int, default=56000, help="target face count (default: 56000)")
a = p.parse_args()

print(f"== 5 reduce and clean (target: {a.faces} faces)", flush=True)

def clean(m):
    m.remove_duplicate_faces()
    m.repair_non_manifold_edges()
    m.remove_small_connected_components(1e-5)
    m.fill_holes(max_hole_perimeter=3e-2)

def reduce_mesh(dense_path, out):
    started = time.time()
    m = trimesh.load(dense_path, process=False)
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
    if os.environ.get("KEEP_LARGEST_COMPONENT") == "1":
        reduced.update_faces(reduced.unique_faces())
        reduced.update_faces(reduced.nondegenerate_faces())
        parts = reduced.split(only_watertight=False, repair=False)
        reduced = max(parts, key=lambda part: part.area)
        trimesh.repair.fill_holes(reduced)
        reduced.remove_unreferenced_vertices()
        if not reduced.is_watertight or not reduced.is_winding_consistent:
            raise RuntimeError("Largest component is not a closed, consistently oriented surface")
    reduced.export(out)
    print(f"reduced to {len(v)} vertices, {len(f)} faces in {time.time() - started:.1f}s -> {out}", flush=True)

run_jobs(a.batch, (a.dense, a.out), reduce_mesh)
