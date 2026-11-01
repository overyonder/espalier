"""Bake saved paintings and package the same mesh, UVs and base PBR material."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path
import nvdiffrast.torch as dr
from paint_base import load_paint_base

sys.path.insert(0, str(Path(__file__).with_name('resident-depth')))
from batch import load_geometry
from resident import camera_matrices
from resident_vk import bake

parser = argparse.ArgumentParser()
parser.add_argument('base', type=Path)
parser.add_argument('cameras', type=Path)
parser.add_argument('paintings', type=Path)
parser.add_argument('bake', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--base-material', type=Path)
args = parser.parse_args()
cameras = json.loads(args.cameras.read_text())
vertices, normals, uv, faces, flat = load_geometry(args.base, True)
base_colour = load_paint_base(args.base, args.base_material).visual.material.baseColorTexture
future = Future()
future.set_result((vertices, normals, uv, faces))
args.bake.mkdir(parents=True, exist_ok=True)
bake(future, camera_matrices(cameras, 'cuda'), cameras,
     [args.paintings / (camera['name'] + '.png') for camera in cameras['cams']],
     args.bake, dr.RasterizeCudaContext(), flat_normals=flat, base_colour=base_colour)
packer = Path(__file__).resolve().parents[1] / 'Intermediate/surfaces/pack-generated'
pending = args.output.with_suffix('.pending.glb')
pending.unlink(missing_ok=True)
subprocess.run([str(packer), str(args.bake/'sparse-uv.ply'), str(args.bake/'atlas-bc7.ktx'),
                str(args.bake/'bake-timings.json'), str(pending), str(args.base)], check=True)
pending.replace(args.output)
print('BAKED', args.output, flush=True)
