"""Bake saved paintings and package the same mesh, UVs and base PBR material."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path
import nvdiffrast.torch as dr
from paint_base import load_paint_base
from PIL import Image

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
parser.add_argument('--base-colour', type=Path)
parser.add_argument('--colour-match-views', type=Path)
args = parser.parse_args()
cameras = json.loads(args.cameras.read_text())
vertices, normals, uv, faces, flat = load_geometry(args.base, True)
base_colour = load_paint_base(args.base, args.base_material).visual.material.baseColorTexture
if args.base_colour:
    with Image.open(args.base_colour) as image:
        base_colour = image.copy()
future = Future()
future.set_result((vertices, normals, uv, faces))
args.bake.mkdir(parents=True, exist_ok=True)
if args.colour_match_views:
    original = args.bake / 'base-original.png'
    corrected = args.bake / 'base-colour-matched.png'
    base_colour.save(original)
    matcher = Path(__file__).resolve().parents[1] / 'Intermediate/surfaces/match-base-colour'
    subprocess.run([str(matcher), str(original), str(args.cameras), str(args.colour_match_views),
                    str(args.paintings), str(corrected), str(args.bake / 'base-colour-match.toml')], check=True)
    with Image.open(corrected) as image:
        base_colour = image.copy()
bake(future, camera_matrices(cameras, 'cuda'), cameras,
     [args.paintings / (camera['name'] + '.png') for camera in cameras['cams']],
     args.bake, dr.RasterizeCudaContext(), flat_normals=flat, base_colour=base_colour)
packer = Path(__file__).resolve().parents[1] / 'Intermediate/surfaces/pack-generated'
pending = args.output.with_suffix('.pending.glb')
pending.unlink(missing_ok=True)
subprocess.run([str(packer), str(args.bake/'sparse-uv.ply'), str(args.bake/'atlas-bc7.ktx'),
                str(args.bake/'bake-timings.json'), str(pending), str(args.base_material or args.base)], check=True)
pending.replace(args.output)
print('BAKED', args.output, flush=True)
