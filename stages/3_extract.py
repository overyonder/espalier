#!/usr/bin/env python3
"""Stage 3: Extract Dense Surface. Extracts dense manifold mesh from field tensor via GPU Dual Marching Cubes.

  python3 stages/3_extract.py <field.pt> <out_dense.ply>
"""
import argparse, os, sys, time

p = argparse.ArgumentParser()
p.add_argument("field", nargs="?", help="input field tensor (.pt)")
p.add_argument("out", nargs="?", help="output dense mesh (.ply)")
p.add_argument("--batch", help="tab-separated table of the positional paths, holding models across its rows")
a = p.parse_args()

from jobs import run_jobs
import torch, trimesh
from skimage import measure
from diso import DiffDMC
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shape_root = os.environ.get("ESPALIER_SHAPE_ROOT", os.path.join(here, "src", "Hunyuan3D-2.1"))
sys.path.insert(0, os.path.join(shape_root, "hy3dshape"))
from hy3dshape.models.autoencoders.surface_extractors import center_vertices

def extract(src, out, extractor):
    kept = torch.load(src, map_location='cpu'); k = kept['kwargs']
    started = time.time()
    # This GPU copy belongs to extraction; scale it in place so large grids need no full-volume temporaries.
    sdf = kept['grid_logits'][0].to(device='cuda', dtype=torch.float32).contiguous()
    sdf.neg_().div_(k['octree_resolution'])
    vertices, faces = extractor(sdf, deform=None, return_quads=False, normalize=True)
    dmc_vertices = center_vertices(vertices).detach().cpu().numpy()
    dmc_faces = faces.detach().cpu().numpy()
    del sdf, vertices, faces
    torch.cuda.empty_cache()
    seconds = time.time() - started
    v, _, _, _ = measure.marching_cubes(kept['grid_logits'][0].numpy(), k['mc_level'], method='lewiner')
    v = v / (k['octree_resolution'] + 1) * (2 * k['bounds']) - k['bounds']
    lo, hi = v.min(0), v.max(0); dlo, dhi = dmc_vertices.min(0), dmc_vertices.max(0)
    scale = ((hi - lo) / (dhi - dlo)).mean()
    mesh = trimesh.Trimesh((dmc_vertices - (dlo + dhi) / 2) * scale + (lo + hi) / 2, dmc_faces, process=False)
    print(f'dmc {seconds:.1f}s: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces -> {out}', flush=True)
    mesh.export(out)

extractor = DiffDMC(dtype=torch.float32).cuda()
run_jobs(a.batch, (a.field, a.out), lambda field, out: extract(field, out, extractor))
