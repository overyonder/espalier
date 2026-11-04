#!/usr/bin/env python3
"""Stage 7: Base PBR Texture. Generates base colour, metallic, and roughness maps via TRELLIS.2 texturing pipeline.

  python3 stages/7_albedo.py <cutout.png> <refined.ply> <unwrapped.glb> <out_base.glb> [--texture 4096] [--dinov3 <repo>]
"""
import argparse, os, sys, time
import numpy as np, torch, trimesh, cv2, flex_gemm, nvdiffrast.torch as dr
from PIL import Image
from scipy.spatial import cKDTree
from jobs import run_jobs

p = argparse.ArgumentParser()
p.add_argument("cutout", nargs="?", help="input cutout PNG")
p.add_argument("dense", nargs="?", help="input refined or dense mesh (.ply)")
p.add_argument("unwrapped", nargs="?", help="input unwrapped model (.glb)")
p.add_argument("out", nargs="?", help="output base textured model (.glb)")
p.add_argument("--batch", help="tab-separated table of the positional paths, holding models across its rows")
p.add_argument("--texture", type=int, default=4096, help="texture atlas size (default: 4096)")
p.add_argument("--dinov3", default=None, help="alternative DINOv3 repository")
p.add_argument("--guidance", type=float, help="classifier-free guidance strength (default: the pipeline's)")
p.add_argument("--seed", type=int, default=42, help="texture sampling seed (default: 42)")
a = p.parse_args()

from cumesh import cuBVH
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src", "TRELLIS.2"))
from trellis2.modules import image_feature_extractor
from trellis2.pipelines import Trellis2TexturingPipeline
if a.dinov3:
    original_init = image_feature_extractor.DinoV3FeatureExtractor.__init__
    image_feature_extractor.DinoV3FeatureExtractor.__init__ = lambda self, model_name, **kw: original_init(self, a.dinov3, **kw)

torch.set_grad_enabled(False)
pipeline = Trellis2TexturingPipeline.from_pretrained('microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
# FlexGEMM checks parameter flags even under no_grad; frozen weights avoid allocating its training-only neighbour maps.
for model in pipeline.models.values():
    model.requires_grad_(False)
pipeline.cuda()

def timed(name, work):
    torch.cuda.synchronize(); begun = time.time(); result = work(); torch.cuda.synchronize()
    print(f'  {name}: {time.time() - begun:.1f}s', flush=True); return result

def generate_texture(image_path, resolution, texture_size, shape_path, unwrapped_path, out):
    started = time.time()
    shape = trimesh.load(shape_path, force='mesh', process=False)
    unwrapped_scene = trimesh.load(unwrapped_path, force='scene', process=False)
    if len(unwrapped_scene.geometry) != 1:
        raise ValueError('Base texturing requires one unwrapped primitive')
    unwrapped = next(iter(unwrapped_scene.geometry.values()))
    # The pipeline's normalisation, applied to each mesh by its own box (the unwrapped mesh may already be normalised, as
    # TRELLIS.2's export is): centre and scale into the unit cube, then y up to z.
    def normalise(v):
        lo, hi = v.min(0), v.max(0); centre, scale = (lo + hi) / 2, 0.99999 / (hi - lo).max()
        return np.stack([(v[:, 0] - centre[0]) * scale, -(v[:, 2] - centre[2]) * scale, (v[:, 1] - centre[1]) * scale], 1)
    shape_normalised = trimesh.Trimesh(normalise(np.asarray(shape.vertices)), shape.faces, process=False)
    image = pipeline.preprocess_image(Image.open(image_path))
    cond = timed('image conditioning', lambda: pipeline.get_cond([image], 512 if resolution == 512 else 1024))
    torch.manual_seed(a.seed)
    shape_latent = timed('voxelise and encode the shape mesh', lambda: pipeline.encode_shape_slat(shape_normalised, resolution))
    texture = timed('generate texture latent', lambda: pipeline.sample_tex_slat(cond, pipeline.models[f'tex_slat_flow_model_{resolution}'], shape_latent, {} if a.guidance is None else {'guidance_strength': a.guidance}))
    del cond, shape_latent
    torch.cuda.empty_cache()
    voxels = timed('decode PBR voxels', lambda: pipeline.decode_tex_slat(texture))
    # The atlas samples only decoded features and coordinates. Release decoder
    # topology caches and allocator blocks before nvdiffrast allocates outside
    # PyTorch's caching allocator.
    del texture
    voxels.clear_spatial_cache()
    torch.cuda.empty_cache()

    def bake():
        vertices = torch.tensor(normalise(np.asarray(unwrapped.vertices)), dtype=torch.float32).cuda()
        faces = torch.tensor(unwrapped.faces, dtype=torch.int32).cuda()
        uvs = np.asarray(unwrapped.visual.uv).copy(); uvs[:, 1] = 1 - uvs[:, 1]
        uv_clip = torch.tensor(np.c_[uvs * 2 - 1, np.zeros(len(uvs)), np.ones(len(uvs))], dtype=torch.float32).cuda()[None].contiguous()
        raster, _ = dr.rasterize(dr.RasterizeCudaContext(), uv_clip, faces, (texture_size, texture_size))
        mask = raster[0, ..., 3] > 0
        points = dr.interpolate(vertices[None], raster, faces)[0][0][mask]
        surface_vertices = torch.as_tensor(np.asarray(shape_normalised.vertices), dtype=torch.float32, device='cuda')
        surface_faces = torch.as_tensor(np.asarray(shape_normalised.faces), dtype=torch.long, device='cuda')
        surface = cuBVH(shape_normalised.vertices, shape_normalised.faces)
        projected = torch.empty_like(points)
        for start in range(0, len(points), 262144):
            end = min(start + 262144, len(points))
            _, face_ids, barycentric = surface.unsigned_distance(points[start:end], return_uvw=True)
            triangles = surface_vertices[surface_faces[face_ids]]
            projected[start:end] = (triangles * barycentric[..., None]).sum(1)
        grid = ((projected + 0.5) * resolution).reshape(1, -1, 3)
        sampled = flex_gemm.ops.grid_sample.grid_sample_3d(
            voxels.feats, voxels.coords, shape=torch.Size([*voxels.shape, *voxels.spatial_shape]),
            grid=grid, mode='trilinear')[0]
        weights = flex_gemm.ops.grid_sample.grid_sample_3d(
            torch.ones_like(voxels.feats[:, :1]), voxels.coords,
            shape=torch.Size([voxels.shape[0], 1, *voxels.spatial_shape]),
            grid=grid, mode='trilinear')[0]
        sampled /= weights.clamp_min(1e-6)
        missing = weights[:, 0] < 1e-6
        if missing.any():
            voxel_positions = voxels.coords[:, 1:].cpu().numpy()
            _, nearest = cKDTree(voxel_positions).query(grid[0, missing].cpu().numpy(), workers=-1)
            sampled[missing] = voxels.feats[torch.as_tensor(nearest, device='cuda')]
        print(f'  continuous surface bake: {len(points)} texels, {int(missing.sum())} nearest-voxel fallbacks', flush=True)
        attrs = torch.zeros(texture_size, texture_size, voxels.shape[1], device='cuda')
        attrs[mask] = sampled
        layout, empty = pipeline.pbr_attr_layout, (~mask).cpu().numpy().astype(np.uint8)
        channel = lambda name: np.clip(attrs[..., layout[name]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        base_color = cv2.inpaint(channel('base_color'), empty, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(channel('metallic'), empty, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(channel('roughness'), empty, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(channel('alpha'), empty, 1, cv2.INPAINT_TELEA)[..., None]
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], -1)),
                                                       metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], -1)),
                                                       metallicFactor=1.0, roughnessFactor=1.0, alphaMode='OPAQUE', doubleSided=True)
        # Changing the material must retain the unwrapped mesh's stored normals.
        unwrapped.visual.material = material
        unwrapped.export(out + '.partial.glb')
        os.replace(out + '.partial.glb', out)
    timed('bake into the unwrapped atlas', bake)
    print(f'total {time.time() - started:.1f}s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB -> {out}', flush=True)


def texture_job(cutout, dense, unwrapped, out):
    generate_texture(cutout, 1024, a.texture, dense, unwrapped, out)
    torch.cuda.empty_cache()

run_jobs(a.batch, (a.cutout, a.dense, a.unwrapped, a.out), texture_job)
