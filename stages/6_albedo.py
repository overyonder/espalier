#!/usr/bin/env python3
"""Stage 6: Base PBR Texture. Generates base colour, metallic, and roughness maps via TRELLIS.2 texturing pipeline.

  python3 stages/6_albedo.py <cutout.png> <dense.ply> <unwrapped.glb> <out_base.glb> [--texture 4096] [--dinov3 <repo>]
"""
import argparse, os, sys, time
import numpy as np, torch, trimesh, cv2, flex_gemm, nvdiffrast.torch as dr
from PIL import Image
from scipy.spatial import cKDTree

p = argparse.ArgumentParser()
p.add_argument("cutout", help="input cutout PNG")
p.add_argument("dense", help="input dense mesh (.ply)")
p.add_argument("unwrapped", help="input unwrapped model (.glb)")
p.add_argument("out", help="output base textured model (.glb)")
p.add_argument("--texture", type=int, default=4096, help="texture atlas size (default: 4096)")
p.add_argument("--dinov3", default=None, help="alternative DINOv3 repository")
a = p.parse_args()

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src", "TRELLIS.2"))

from trellis2.modules import image_feature_extractor
from trellis2.pipelines import Trellis2TexturingPipeline

print(f"== 6 albedo & PBR (TRELLIS.2 1024 voxel / {a.texture} texture)", flush=True)

if a.dinov3:
    orig = image_feature_extractor.DinoV3FeatureExtractor.__init__
    image_feature_extractor.DinoV3FeatureExtractor.__init__ = lambda self, model_name, **kw: orig(self, a.dinov3, **kw)

torch.set_grad_enabled(False)
pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
pipeline.cuda()

started = time.time()
shape = trimesh.load(a.dense, force="mesh", process=False)
unwrapped = trimesh.load(a.unwrapped, force="mesh", process=False)

def normalise(v):
    lo, hi = v.min(0), v.max(0)
    centre, scale = (lo + hi) / 2, 0.99999 / (hi - lo).max()
    return np.stack([(v[:, 0] - centre[0]) * scale, -(v[:, 2] - centre[2]) * scale, (v[:, 1] - centre[1]) * scale], 1)

shape_normalised = trimesh.Trimesh(normalise(np.asarray(shape.vertices)), shape.faces, process=False)
image = pipeline.preprocess_image(Image.open(a.cutout))
cond = pipeline.get_cond([image], 1024)
shape_latent = pipeline.encode_shape_slat(shape_normalised, 1024)
texture = pipeline.sample_tex_slat(cond, pipeline.models["tex_slat_flow_model_1024"], shape_latent, {})
voxels = pipeline.decode_tex_slat(texture)

vertices = torch.tensor(normalise(np.asarray(unwrapped.vertices)), dtype=torch.float32).cuda()
faces = torch.tensor(unwrapped.faces, dtype=torch.int32).cuda()
uvs = np.asarray(unwrapped.visual.uv).copy()
uvs[:, 1] = 1 - uvs[:, 1]
uv_clip = torch.tensor(np.c_[uvs * 2 - 1, np.zeros(len(uvs)), np.ones(len(uvs))], dtype=torch.float32).cuda()[None].contiguous()
raster, _ = dr.rasterize(dr.RasterizeCudaContext(), uv_clip, faces, (a.texture, a.texture))
mask = raster[0, ..., 3] > 0
points = dr.interpolate(vertices[None], raster, faces)[0][0][mask].cpu().numpy()
_, nearest = cKDTree(shape_normalised.vertices).query(points, workers=-1)
points = torch.tensor(shape_normalised.vertices[nearest], dtype=torch.float32).cuda()

attrs = torch.zeros(a.texture, a.texture, voxels.shape[1], device="cuda")
attrs[mask] = flex_gemm.ops.grid_sample.grid_sample_3d(
    voxels.feats, voxels.coords, shape=torch.Size([*voxels.shape, *voxels.spatial_shape]),
    grid=((points + 0.5) * 1024).reshape(1, -1, 3), mode="trilinear")

layout, empty = pipeline.pbr_attr_layout, (~mask).cpu().numpy().astype(np.uint8)
channel = lambda name: np.clip(attrs[..., layout[name]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
base_color = cv2.inpaint(channel("base_color"), empty, 3, cv2.INPAINT_TELEA)
metallic = cv2.inpaint(channel("metallic"), empty, 1, cv2.INPAINT_TELEA)[..., None]
roughness = cv2.inpaint(channel("roughness"), empty, 1, cv2.INPAINT_TELEA)[..., None]
material = trimesh.visual.material.PBRMaterial(
    baseColorTexture=Image.fromarray(base_color),
    metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], -1)),
    metallicFactor=1.0, roughnessFactor=1.0)

trimesh.Trimesh(unwrapped.vertices, unwrapped.faces, process=False,
                visual=trimesh.visual.TextureVisuals(uv=np.asarray(unwrapped.visual.uv), material=material)).export(a.out)
print(f"baked base texture in {time.time() - started:.1f}s -> {a.out}", flush=True)
