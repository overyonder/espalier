"""Render textured or camera-normal paint inputs using the exact projection cameras."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import nvdiffrast.torch as dr
import torch
from paint_base import load_paint_base
from PIL import Image

sys.path.insert(0, str(Path(__file__).with_name('resident-depth')))
from batch import fit_cameras
from resident import camera_matrices, fit_lens_to_silhouette

parser = argparse.ArgumentParser()
parser.add_argument('mesh', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('cameras', type=Path)
parser.add_argument('--base', choices=['trellis', 'normals'], default='trellis')
parser.add_argument('--base-material', type=Path)
parser.add_argument('--reframe', nargs='+', default=[], help='fit these saved cameras to the silhouette and render them again')
args = parser.parse_args()
mesh = load_paint_base(args.mesh, args.base_material)
vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
faces = torch.tensor(mesh.faces, dtype=torch.int32, device='cuda')
world = torch.stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1], torch.ones_like(vertices[:, 0])), 1)
if args.cameras.is_file():
    cameras = json.loads(args.cameras.read_text())
else:
    cameras = fit_cameras(vertices)
    args.cameras.write_text(json.dumps(cameras, indent=2))
if args.reframe:
    points = world[:, :3].cpu().numpy().astype('f8')
    for camera in cameras['cams']:
        if camera['name'] in args.reframe:
            fit_lens_to_silhouette(camera, points)
            (args.output / (camera['name'] + '.png')).unlink(missing_ok=True)
    args.cameras.write_text(json.dumps(cameras, indent=2))
matrices = camera_matrices(cameras, 'cuda')
if args.base == 'trellis':
    coordinates = np.array(mesh.visual.uv, dtype='f4')
    coordinates[:, 1] = 1 - coordinates[:, 1]
    coordinates = torch.tensor(coordinates, device='cuda')
    texture = torch.tensor(np.array(mesh.visual.material.baseColorTexture.convert('RGB')), dtype=torch.float32, device='cuda')[None] / 255
else:
    normals = torch.tensor(np.array(mesh.vertex_normals, copy=True), dtype=torch.float32, device='cuda')
    normals = torch.stack((normals[:, 0], -normals[:, 2], normals[:, 1]), 1)
args.output.mkdir(parents=True, exist_ok=True)
context = dr.RasterizeCudaContext()
for index, camera in enumerate(cameras['cams']):
    output = args.output / (camera['name'] + '.png')
    if output.is_file():
        continue
    raster, _ = dr.rasterize(context, (world @ matrices[index].T)[None].contiguous(), faces,
                             resolution=(cameras['res'], cameras['res']), grad_db=False)
    if args.base == 'trellis':
        uv, _ = dr.interpolate(coordinates[None], raster, faces)
        colour = dr.texture(texture, uv, filter_mode='linear', boundary_mode='clamp')
    else:
        rotation = torch.tensor(np.array(camera['matrix'])[:3, :3], dtype=torch.float32, device='cuda')
        colour, _ = dr.interpolate((normals @ rotation)[None].contiguous(), raster, faces)
        colour = torch.nn.functional.normalize(colour, dim=-1) * .5 + .5
    colour = torch.where(raster[..., 3:4] > 0, colour, 0)
    Image.fromarray((colour[0].flip(0).clamp(0, 1) * 255).byte().cpu().numpy()).save(output)
    print('RENDERED', camera['name'], cameras['res'], args.base, flush=True)
