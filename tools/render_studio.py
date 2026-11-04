"""Render textured models from the saved paint cameras under clean studio lighting, so textures compare as they will look lit."""
import argparse
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
from paint_base import load_paint_base
from PIL import Image

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'src/TRELLIS.2'), str(Path(__file__).with_name('resident-depth'))]
from resident import camera_matrices
from trellis2.renderers.pbr_mesh_renderer import EnvMap

parser = argparse.ArgumentParser()
parser.add_argument('model', type=Path)
parser.add_argument('cameras', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--views', nargs='+')
parser.add_argument('--base-colour', type=Path, help='use this atlas in place of the model\'s colour')
parser.add_argument('--environment', type=Path, default=root / 'Intermediate/surfaces/studio.hdr', help='a Radiance image: this OpenCV build reads no EXR')
parser.add_argument('--roughness', type=float, default=0.6, help='fixed for every model, so renders differ in colour alone')
parser.add_argument('--exposure', type=float, default=1.0)
parser.add_argument('--match-luma', type=float, help='set the exposure so the first view has this mean sRGB luma, and print it')
parser.add_argument('--background', type=float, default=0.66)
args = parser.parse_args()
LUMA = torch.tensor([0.2126, 0.7152, 0.0722], device='cuda')
SUPERSAMPLE = 2

mesh = load_paint_base(args.model)
if args.base_colour:
    mesh.visual.material.baseColorTexture = Image.open(args.base_colour)
vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device='cuda')
faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device='cuda')
normals = torch.tensor(np.array(mesh.vertex_normals, copy=True), dtype=torch.float32, device='cuda')
coordinates = np.array(mesh.visual.uv, dtype='f4')
coordinates[:, 1] = 1 - coordinates[:, 1]
coordinates = torch.tensor(coordinates, device='cuda')
texture = torch.tensor(np.array(mesh.visual.material.baseColorTexture.convert('RGB')), dtype=torch.float32, device='cuda')[None] / 255
cameras = json.loads(args.cameras.read_text())
matrices = camera_matrices(cameras, 'cuda')
world = torch.stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1], torch.ones_like(vertices[:, 0])), 1)
lighting = torch.tensor(cv2.cvtColor(cv2.imread(str(args.environment), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device='cuda')
# White balance by solid angle, because a tinted light would read as a difference in the texture.
latitude = (torch.arange(lighting.shape[0], device='cuda') + 0.5) / lighting.shape[0] * torch.pi - torch.pi / 2
area = torch.cos(latitude)[:, None, None]
tint = (lighting * area).sum((0, 1)) / (area.sum() * lighting.shape[1])
environment = EnvMap(lighting / (tint / tint.mean()))
context = dr.RasterizeCudaContext()
args.output.mkdir(parents=True, exist_ok=True)


def to_srgb(linear):
    linear = linear.clamp(0, 1)
    return torch.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear.clamp_min(1e-6) ** (1 / 2.4) - 0.055)


exposure = args.exposure
for index, camera in enumerate(cameras['cams']):
    if args.views and camera['name'] not in args.views:
        continue
    size = cameras['res'] * SUPERSAMPLE
    raster, _ = dr.rasterize(context, (world @ matrices[index].T)[None].contiguous(), faces, resolution=(size, size), grad_db=False)
    uv, _ = dr.interpolate(coordinates[None], raster, faces)
    albedo = dr.texture(texture, uv, filter_mode='linear', boundary_mode='clamp') ** 2.2
    position, _ = dr.interpolate(vertices[None], raster, faces)
    normal, _ = dr.interpolate(normals[None], raster, faces)
    normal = torch.nn.functional.normalize(normal, dim=-1)
    # The cameras are stored Z up, and the model and environment are Y up.
    eye = np.array(camera['matrix'])[:3, 3]
    view = torch.tensor([eye[0], eye[2], -eye[1]], dtype=torch.float32, device='cuda')[None, None, None]
    material = torch.cat((torch.zeros_like(albedo[..., :1]), torch.full_like(albedo[..., :1], args.roughness), torch.zeros_like(albedo[..., :1])), -1)
    shaded = environment.shade(position, normal, albedo, material, view, specular=True)
    covered = raster[..., 3:4] > 0
    if args.match_luma is not None and exposure == args.exposure:
        low, high = 0.01, 100.0
        for _ in range(30):
            exposure = (low * high) ** 0.5
            luma = (to_srgb(shaded[covered[..., 0]] * exposure) * LUMA).sum(-1).mean()
            low, high = (exposure, high) if luma < args.match_luma else (low, exposure)
        print('EXPOSURE', f'{exposure:.4f}', flush=True)
    colour = torch.where(covered, to_srgb(shaded * exposure), args.background)
    image = (colour[0].flip(0) * 255).byte().cpu().numpy()
    Image.fromarray(image).resize((cameras['res'], cameras['res']), Image.LANCZOS).save(args.output / (camera['name'] + '.png'))
    print('RENDERED', camera['name'], flush=True)
