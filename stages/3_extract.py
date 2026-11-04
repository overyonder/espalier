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
import numpy as np, torch, trimesh
from scipy import ndimage
from skimage import measure
from diso import DiffDMC
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shape_root = os.path.join(here, "src", "Hunyuan3D-2.1")
sys.path.insert(0, os.path.join(shape_root, "hy3dshape"))
from hy3dshape.models.autoencoders.surface_extractors import center_vertices

def fill_undecoded(grid, level):
    """Give each undecoded region the side of the surface its decoded neighbours are on, so it adds no surface."""
    undecoded = np.isnan(grid)
    if not undecoded.any():
        return grid
    regions, count = ndimage.label(undecoded)
    inside, outside = np.zeros((2, count + 1), np.int64)
    for axis in range(3):
        for here_slice, there_slice in ((slice(0, -1), slice(1, None)), (slice(1, None), slice(0, -1))):
            near = [slice(None)] * 3; far = [slice(None)] * 3
            near[axis] = here_slice; far[axis] = there_slice
            near = tuple(near); far = tuple(far)
            edge = undecoded[near] & ~undecoded[far]
            labels = regions[near][edge]; positive = grid[far][edge] > level
            inside += np.bincount(labels[positive], minlength=count + 1)
            outside += np.bincount(labels[~positive], minlength=count + 1)
    grid[undecoded] = np.where(inside > outside, 10000.0, -10000.0)[regions[undecoded]]
    return grid

def extract(src, out, extractor):
    kept = torch.load(src, map_location='cpu'); k = kept['kwargs']
    started = time.time()
    grid = fill_undecoded(kept['grid_logits'][0].numpy().astype(np.float32), k['mc_level'])
    resolution = grid.shape[0] - 1
    # This GPU copy belongs to extraction; scale it in place so large grids need no full-volume temporaries.
    sdf = torch.from_numpy(grid).to('cuda').contiguous()
    sdf.neg_().div_(resolution)
    vertices, faces = extractor(sdf, deform=None, return_quads=False, normalize=True)
    dmc_vertices = center_vertices(vertices).detach().cpu().numpy()
    dmc_faces = faces.detach().cpu().numpy()
    del sdf, vertices, faces
    torch.cuda.empty_cache()
    seconds = time.time() - started
    v, _, _, _ = measure.marching_cubes(grid, k['mc_level'], method='lewiner')
    v = v / (resolution + 1) * (2 * k['bounds']) - k['bounds']
    lo, hi = v.min(0), v.max(0); dlo, dhi = dmc_vertices.min(0), dmc_vertices.max(0)
    scale = ((hi - lo) / (dhi - dlo)).mean()
    mesh = trimesh.Trimesh((dmc_vertices - (dlo + dhi) / 2) * scale + (lo + hi) / 2, dmc_faces, process=False)
    print(f'dmc {seconds:.1f}s: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces -> {out}', flush=True)
    mesh.export(out)

extractor = DiffDMC(dtype=torch.float32).cuda()
run_jobs(a.batch, (a.field, a.out), lambda field, out: extract(field, out, extractor))
