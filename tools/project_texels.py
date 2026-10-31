# Projects one painted view into an albedo texture: every texel of the atlas is placed on the surface, projected
# into the view's camera and, where the camera's depth buffer shows that texel's point as the nearest surface, takes the
# colour of the one pixel it lands on. Texels an earlier projection already coloured keep their colour.
#   project_texels.py <base.glb | previous.glb> <painted view.png> <azimuth[:elevation] degrees> <texture size>
#       <out prefix> [facing views]
# With facing views (comma-separated azimuth:elevation, including this view's), the view instead repaints every texel it
# sees whose surface faces this camera more squarely than any of the other listed cameras, replacing an earlier colour.
# A previous projection's .glb has its texture and claimed texels beside it as <previous>-texture.png and
# <previous>-mask.png; a starting albedo's .glb starts from its own base colour. Writes <out>.glb, <out>-texture.png and
# <out>-mask.png. The mesh is Y up, and the view is render/render_views.py's view of it at that azimuth: orthographic
# and level, framed so the mesh's height fills FILL of the square image, or with an elevation (its VIEWS form) tilted
# and framed so the bounding box's diagonal fills FILL.
import os, sys, numpy as np, torch, trimesh, nvdiffrast.torch as dr
from PIL import Image

FILL = 0.88
view_angles = lambda text: tuple(float(x) for x in (text + ':0').split(':')[:2])
mesh_path, view_path, (azimuth, elevation), size, out = sys.argv[1], sys.argv[2], view_angles(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sphere_framing = ':' in sys.argv[3]
facing = [view_angles(x) for x in sys.argv[6].split(',')] if len(sys.argv) > 6 else None

m = trimesh.load(mesh_path, process=False, force='mesh')
vertices, faces, uvs = (torch.tensor(np.asarray(a), dtype=d).cuda().contiguous() for a, d in
                        ((m.vertices, torch.float32), (m.faces, torch.int32), (m.visual.uv, torch.float32)))
previous = mesh_path[:-len('.glb')]
if os.path.exists(f'{previous}-texture.png'):        # an earlier projection's texture and claimed texels
    texture = np.array(Image.open(f'{previous}-texture.png').convert('RGB'))
    claimed = np.array(Image.open(f'{previous}-mask.png')) > 127
else:                                                 # a starting albedo: its own colour, nothing claimed yet
    texture = np.array(m.visual.material.baseColorTexture.convert('RGB').resize((size, size)))
    claimed = np.zeros((size, size), dtype=bool)

# Blender's frame for the stood-up mesh: x stays, height (raw y) is z, and raw z points back toward the front camera.
# render_views.py's camera at this view looks along `forward` at the bounding box's centre, with `right` across the image
# and `up` up the image; points are expressed here as image x, depth and image height, so the rest is the front
# view's arithmetic.
stood = torch.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], 1)
box_lo, box_hi = stood.min(0).values, stood.max(0).values
a, e = np.radians(azimuth), np.radians(elevation)
right = torch.tensor([np.cos(a), np.sin(a), 0.0], dtype=torch.float32).cuda()
forward = torch.tensor([-np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), -np.sin(e)], dtype=torch.float32).cuda()
up = torch.tensor([-np.sin(a) * np.sin(e), np.cos(a) * np.sin(e), np.cos(e)], dtype=torch.float32).cuda()
blender = torch.stack([(stood - (box_lo + box_hi) / 2) @ right, stood @ forward, (stood - (box_lo + box_hi) / 2) @ up], 1).contiguous()
lo, hi = blender.min(0).values, blender.max(0).values
centre = torch.zeros(3).cuda()
# The orthographic camera's width and height in world units: the mesh's height, or the box's diagonal for tilted views.
scale = float((box_hi - box_lo).norm() if sphere_framing else box_hi[2] - box_lo[2]) / FILL

context = dr.RasterizeCudaContext()
# Texel positions: rasterise the mesh in UV space and interpolate each texel's point on the surface.
uv_clip = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], 1)[None]
texel_raster, _ = dr.rasterize(context, uv_clip, faces, (size, size))
texel_points, _ = dr.interpolate(blender[None].contiguous(), texel_raster, faces)
texel_points, covered = texel_points[0], texel_raster[0, ..., 3] > 0

# The camera's depth buffer at the painted view's resolution: nearest surface depth (Blender y) per pixel.
view = np.asarray(Image.open(view_path).convert('RGB'))
height_px, width_px = view.shape[:2]
camera_clip = torch.stack([(blender[:, 0] - centre[0]) / (scale / 2), (blender[:, 2] - centre[2]) / (scale / 2),
                           (blender[:, 1] - lo[1]) / float(hi[1] - lo[1] + 1e-9) * 2 - 1, torch.ones_like(blender[:, 0])], 1)[None]
camera_raster, _ = dr.rasterize(context, camera_clip, faces, (height_px, width_px))
camera_depth, _ = dr.interpolate(blender[None, :, 1:2].contiguous(), camera_raster, faces)
camera_depth, camera_hit = camera_depth[0, ..., 0], camera_raster[0, ..., 3] > 0

# Each texel's pixel. nvdiffrast's row 0 is the bottom of the image, so rows are flipped to match the PNG.
column = ((texel_points[..., 0] - centre[0]) / scale * width_px + width_px / 2).floor().long()
row_from_bottom = ((texel_points[..., 2] - centre[2]) / scale * height_px + height_px / 2).floor().long()
inside = covered & (column >= 0) & (column < width_px) & (row_from_bottom >= 0) & (row_from_bottom < height_px)
c, r = column.clamp(0, width_px - 1), row_from_bottom.clamp(0, height_px - 1)
tolerance = float(hi[1] - lo[1]) * 0.003                    # depth agreement for the nearest surface, 0.3% of the depth
visible = inside & camera_hit[r, c] & ((texel_points[..., 1] - camera_depth[r, c]).abs() <= tolerance)

# Texture rows run downward while UV v runs upward, so the stored texture and mask are flipped to UV order here.
texture_gpu = torch.from_numpy(np.ascontiguousarray(texture[::-1])).cuda()
claimed_gpu = torch.from_numpy(np.ascontiguousarray(claimed[::-1])).cuda()
if facing is None:
    new = visible & ~claimed_gpu
else:
    # Each texel's triangle normal in the stood-up frame, and the direction toward each listed camera.
    normals = torch.cross(stood[faces[:, 1].long()] - stood[faces[:, 0].long()], stood[faces[:, 2].long()] - stood[faces[:, 0].long()], dim=1)
    normals = torch.nn.functional.normalize(normals, dim=1)
    texel_normals = normals[(texel_raster[0, ..., 3].long() - 1).clamp(min=0)]
    toward = torch.tensor([[np.cos(np.radians(v)) * np.sin(np.radians(z)), -np.cos(np.radians(v)) * np.cos(np.radians(z)), np.sin(np.radians(v))]
                           for z, v in facing], dtype=torch.float32).cuda()
    best = (texel_normals @ toward.T).argmax(-1)
    new = visible & (best == facing.index((azimuth, elevation)))
picked = torch.from_numpy(view).cuda()[height_px - 1 - r, c]
texture_gpu[new] = picked[new]
claimed_gpu |= new
texture = texture_gpu.cpu().numpy()[::-1]
claimed = claimed_gpu.cpu().numpy()[::-1]
print(f'texels on the surface: {int(covered.sum())}, seen by this view: {int(visible.sum())}, newly painted: {int(new.sum())}, '
      f'painted in total: {int(claimed.sum())} ({100 * float(claimed.sum()) / float(covered.sum()):.1f}%)', flush=True)

Image.fromarray(np.ascontiguousarray(texture)).save(f'{out}-texture.png')
Image.fromarray(np.ascontiguousarray(claimed.astype(np.uint8) * 255)).save(f'{out}-mask.png')
material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(texture), metallicFactor=0.0, roughnessFactor=0.8)
trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy(), process=False,
                visual=trimesh.visual.TextureVisuals(uv=uvs.cpu().numpy(), material=material)).export(f'{out}.glb')
print('wrote', out, flush=True)
