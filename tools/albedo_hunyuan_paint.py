# Generates the albedo, with metallic and roughness, with Hunyuan3D-Paint 2.1 on the unwrapped mesh. Tencent's pipeline
# paints base colour and metallic-roughness views together from the mesh's normal and position maps and the reference
# image, upscales them with Real-ESRGAN and bakes them into the atlas. Its own remesh is off and its xatlas unwrap is
# skipped, so the texture lands in the unwrapped mesh's atlas. Writes <out dir>/textured.obj with its textures, and a
# GLB copy with all three maps.
#   cd ~/code/models/Hunyuan3D-2.1 && .venv/bin/python <espalier>/tools/albedo_hunyuan_paint.py <unwrapped.glb> \
#       <cutout.png> <out dir> [views 6-9] [view resolution 512|768]
import os, sys, time, types
root = os.getcwd()
sys.path.insert(0, os.path.join(root, 'hy3dpaint'))
# Tencent's mesh utilities import Blender's Python module only to convert their OBJ to GLB; that conversion is done
# here with trimesh instead.
sys.modules['bpy'] = types.ModuleType('bpy')
from utils.torchvision_fix import apply_fix   # Tencent's patch for basicsr's import of a torchvision module since removed
apply_fix()
import trimesh
import textureGenPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

def keep_existing_atlas(mesh):
    mesh = mesh.dump(concatenate=True) if isinstance(mesh, trimesh.Scene) else mesh
    if getattr(mesh.visual, 'uv', None) is None:
        raise ValueError('the mesh needs its atlas')
    return mesh
textureGenPipeline.mesh_uv_wrap = keep_existing_atlas

mesh_path, image_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
views, resolution = (int(sys.argv[4]) if len(sys.argv) > 4 else 6), (int(sys.argv[5]) if len(sys.argv) > 5 else 512)
os.makedirs(out, exist_ok=True)
started = time.time()
config = Hunyuan3DPaintConfig(views, resolution)
# Tencent's paths are relative to different directories: the configuration to the checkout, the upscaler to hy3dpaint.
config.multiview_cfg_path = os.path.join(root, 'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml')
config.realesrgan_ckpt_path = os.path.join(root, 'hy3dpaint/ckpt/RealESRGAN_x4plus.pth')
pipeline = Hunyuan3DPaintPipeline(config)
print(f'models loaded in {time.time() - started:.0f}s', flush=True)
started = time.time()
result = pipeline(mesh_path=mesh_path, image_path=image_path, output_mesh_path=os.path.join(out, 'textured.obj'), use_remesh=False, save_glb=False)
# The GLB copy carries Tencent's metallic and roughness too, packed as glTF's metallic-roughness texture (roughness in
# green, metallic in blue); trimesh's OBJ reader takes only the base colour.
import numpy as np
from PIL import Image
mesh = trimesh.load(result, force='mesh', process=False)
stem = os.path.splitext(result)[0]
metallic, roughness = (np.array(Image.open(f'{stem}_{name}.jpg').convert('L')) for name in ('metallic', 'roughness'))
material = trimesh.visual.material.PBRMaterial(baseColorTexture=mesh.visual.material.image,
                                               metallicRoughnessTexture=Image.fromarray(np.stack([np.zeros_like(metallic), roughness, metallic], -1)),
                                               metallicFactor=1.0, roughnessFactor=1.0)
trimesh.Trimesh(mesh.vertices, mesh.faces, process=False,
                visual=trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)).export(os.path.join(out, 'textured.glb'))
print(f'textured in {time.time() - started:.0f}s -> {result}', flush=True)
