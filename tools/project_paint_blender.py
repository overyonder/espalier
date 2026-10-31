"""Progressive projection painting's Blender side (driven by espalier's progressive painting pass).

  blender -b --python project_paint_blender.py -- piece   <base.glb> <piece.blend>
  blender -b --python project_paint_blender.py -- render  <piece.blend> <dir> [views=orbit] [tiles=1] [res=1024]
  (paint <dir>/views/<name>.png -> <dir>/gen/<name>.png with Codex image editing)
  blender -b --python project_paint_blender.py -- project <piece.blend> <dir> [tex=2048] [match=1] [only=<view>] [blend=average|progressive]
  blender -b --python project_paint_blender.py -- export  <textured.blend> <painted.glb>

render: cameras orbit the piece (8 azimuths at 15 degrees elevation, 4 at 50, one from above), each optionally split into
tiles x tiles sensor-shift cameras with the lens scaled up, so a tile is the same perspective at tiles-times the pixel
density. Each camera writes the current base colour lit flat (views/<name>.png) and the surface position per pixel
(pos/<name>.exr); cams.json holds the cameras.
project: bakes the surface position and normal of every texel, projects each texel into every generated view, keeps it
where that view's position pass sees the same point (visibility), weights it by how squarely and how centrally the view
sees it, optionally colour-matches each view to its render of the base first, averages, falls back to the base where no
view reaches, pads the UV islands, and writes <dir>/base_color.png and <dir>/textured.blend plus its texel density.
blend=progressive (with only=<view>) blends each view in by w^4 / (w^4 + best^4) against a
per-texel best-weight map kept in <dir>, instead of averaging every view at once; sharp=<k> sets that exponent.
Views render in EEVEE; surface position and normal bakes run in Cycles and are cached in <dir>.
"""
import bpy, bmesh, sys, os, json, math
import numpy as np
from mathutils import Vector, Matrix

a = sys.argv[sys.argv.index("--") + 1:]
mode, src, out = a[0], a[1], a[2]
opt = dict(x.split("=") for x in a[3:])
if mode == "piece":     # a starting albedo's GLB as the .blend the painter works on, its base colour image packed in
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=src)
    for image in bpy.data.images: image.pack()
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True); sys.exit(0)
if mode == "export":    # the painted .blend as a GLB
    bpy.ops.wm.open_mainfile(filepath=src); bpy.ops.export_scene.gltf(filepath=out, export_format='GLB'); sys.exit(0)
os.makedirs(out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
with bpy.data.libraries.load(src) as (f, t): t.objects = f.objects
obj = [o for o in t.objects if o.type == 'MESH'][0]
sc = bpy.context.scene; sc.collection.objects.link(obj)
mat = obj.data.materials[0]
base_node = next(n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and
                 any(l.to_socket.name == "Base Color" for l in n.outputs["Color"].links))
base_img = base_node.image
pts = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
lo, hi = pts.min(0), pts.max(0); centre = Vector(((lo + hi) / 2).tolist()); radius = float(np.linalg.norm(hi - lo) / 2)

sc.render.engine = 'CYCLES'; sc.cycles.device = 'GPU'
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = 'CUDA'; prefs.get_devices()
for d in prefs.devices: d.use = True
sc.view_settings.view_transform = 'Standard'; sc.view_settings.look = 'None'
sc.world = bpy.data.worlds.new("w"); sc.world.color = (0, 0, 0)

def emit_material(name, build):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial"); e = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(build(nt), e.inputs["Color"]); nt.links.new(e.outputs[0], o.inputs["Surface"])
    return m

def tex_colour(nt):
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = base_img; t.interpolation = 'Cubic'; return t.outputs["Color"]
# EEVEE renders through half floats and nothing emits below zero, so positions are emitted normalised into 0..1 about
# the piece's centre (sub-millimetre steps for a metre-scale piece) and normals as n / 2 + 0.5
def geo(socket, scale, shift):
    def b(nt):
        m = nt.nodes.new("ShaderNodeVectorMath"); m.operation = 'MULTIPLY_ADD'
        m.inputs[1].default_value = (scale,) * 3; m.inputs[2].default_value = shift
        nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs[socket], m.inputs[0]); return m.outputs[0]
    return b
PS = 1 / (2.2 * radius); PSHIFT = tuple(0.5 - c * PS for c in centre)
def decode_pos(p): return (p - np.array(PSHIFT)) / PS
M_COL, M_POS, M_NRM = emit_material("col", tex_colour), emit_material("pos", geo("Position", PS, PSHIFT)), emit_material("nrm", geo("Normal", 0.5, (0.5, 0.5, 0.5)))
def use(m): obj.data.materials.clear(); obj.data.materials.append(m)

if mode == "render":
    res = int(opt.get("res", 1024)); sc.render.resolution_x = sc.render.resolution_y = res
    tiles = int(opt.get("tiles", 1))
    cams = []
    # 8 azimuths at 15 degrees elevation, 4 at 50, one from above (85)
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315, 45, 135, 225, 315, 0]
    elevations = [15] * 8 + [50] * 4 + [85]
    cams_file = f"{out}/cams.json"
    if not os.path.exists(cams_file):
        for a, e in zip(azimuths, elevations):
            ar, er = math.radians(a), math.radians(e)
            eye = centre + Vector((math.cos(er) * math.sin(ar), -math.cos(er) * math.cos(ar), math.sin(er))) * (radius * 2.8)
            fwd = (centre - eye).normalized(); up = Vector((0, 0, 1))
            right = fwd.cross(up).normalized(); up = right.cross(fwd).normalized()
            mat = Matrix((right, up, -fwd)).to_4x4(); mat.translation = eye
            for ty in range(tiles):
                for tx in range(tiles):
                    shift = ((tx - (tiles - 1) / 2) / tiles, (ty - (tiles - 1) / 2) / tiles)
                    tag = f"_t{tx}{ty}" if tiles > 1 else ""
                    cams.append(dict(name=f"a{a:03d}e{e:02d}{tag}", az=a, el=e, tx=tx, ty=ty,
                                     matrix=[list(r) for r in mat], lens=50.0 * tiles, sensor=36.0, shift=shift))
        os.makedirs(f"{out}/views", exist_ok=True); os.makedirs(f"{out}/pos", exist_ok=True)
        json.dump(dict(radius=radius, centre=list(centre), tiles=tiles, cams=cams), open(cams_file, "w"), indent=2)
    else:
        cams = json.load(open(cams_file))["cams"]
    cam_data = bpy.data.cameras.new("cam"); cam_obj = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam_obj); sc.camera = cam_obj
    cam_data.sensor_width = 36.0
    only = opt.get("only", None)
    for c in cams:
        if only and c["name"] != only: continue
        cam_obj.matrix_world = Matrix(c["matrix"])
        cam_data.lens = c["lens"]; cam_data.shift_x, cam_data.shift_y = c["shift"]
        view_path = f"{out}/views/{c['name']}.png"; pos_path = f"{out}/pos/{c['name']}.exr"
        # base colour in EEVEE
        if not os.path.exists(view_path):
            sc.render.engine = 'BLENDER_EEVEE_NEXT'; use(M_COL)
            sc.render.filepath = view_path; sc.render.image_settings.file_format = 'PNG'
            sc.render.image_settings.color_mode = 'RGBA'
            bpy.ops.render.render(write_still=True)
        # surface position in Cycles (EEVEE doesn't write OpenEXR 32-bit floats properly)
        if not os.path.exists(pos_path):
            sc.render.engine = 'CYCLES'; use(M_POS)
            sc.render.filepath = pos_path; sc.render.image_settings.file_format = 'OPEN_EXR'
            sc.render.image_settings.color_mode = 'RGBA'; sc.render.image_settings.color_depth = '32'
            sc.cycles.samples = 1
            bpy.ops.render.render(write_still=True)
        print("RENDERED", c["name"])

if mode == "project":
    size = int(opt.get("tex", 2048)); match = int(opt.get("match", 1))
    blend = opt.get("blend", "average"); sharp = float(opt.get("sharp", 4.0))
    only = opt.get("only", None)
    res = json.load(open(f"{out}/cams.json"))
    cams, tiles = res["cams"], res["tiles"]

    def bake(socket, m, path):
        if os.path.exists(path): return np.load(path)
        use(m)
        img = bpy.data.images.new(socket, size, size, float_buffer=True, alpha=True)
        tn = m.node_tree.nodes.new("ShaderNodeTexImage"); tn.image = img; m.node_tree.nodes.active = tn
        sc.render.engine = 'CYCLES'; sc.render.bake.use_selected_to_active = False
        sc.cycles.bake_type = 'EMIT'; sc.cycles.samples = 1
        bpy.ops.object.bake(type='EMIT')
        arr = np.array(img.pixels[:], dtype=np.float32).reshape(size, size, 4)
        np.save(path, arr); return arr

    pos = decode_pos(bake("pos", M_POS, f"{out}/pos-bake.npy")[..., :3])
    nrm = bake("nrm", M_NRM, f"{out}/nrm-bake.npy")[..., :3] * 2 - 1
    valid = np.linalg.norm(nrm, axis=-1) > 0.5
    pos, nrm = pos[valid], nrm[valid]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    # Blender image pixels are in [0, 1] sRGB (when read via pixels[:]), but math on them needs linear light
    def srgb_to_lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    def lin_to_srgb(c):
        c = np.clip(c, 0, 1)
        return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)

    base = np.array(base_img.pixels[:], dtype=np.float32).reshape(base_img.size[1], base_img.size[0], 4)[..., :3]
    if base.shape[0] != size:
        # resample base to texture size
        import cv2
        base = cv2.resize(base, (size, size), interpolation=cv2.INTER_AREA)

    def load(path):
        img = bpy.data.images.load(path); arr = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], img.channels)
        bpy.data.images.remove(img); return arr

    acc = np.zeros((len(pos), 3), np.float32); wsum = np.zeros(len(pos), np.float32)
    best = np.zeros(len(pos), np.float32)
    res = cams[0]["lens"] / (50.0 * tiles) * 1024     # view pixel res
    res = int(round(res))
    eps = 0.015 * radius     # visibility tolerance against the surface position pass

    for c in cams:
        if only and c["name"] != only: continue
        g = f"{out}/gen/{c['name']}.png"
        if not os.path.exists(g): continue
        gen = load(g)[..., :3]   # any square size: an upscaled view is sampled at its own resolution
        gres = gen.shape[0]; scale = gres / res
        ren = load(f"{out}/views/{c['name']}.png"); rpos = load(f"{out}/pos/{c['name']}.exr"); rpos[..., :3] = decode_pos(rpos[..., :3])
        alpha = ren[..., 3] > 0.5
        if match:   # per-channel mean and spread of the generated view matched to the base render, over the object
            step = max(1, int(round(scale)))
            small = gen[::step, ::step][:res, :res]
            for k in range(3):
                gm, gs = small[..., k][alpha].mean(), small[..., k][alpha].std() + 1e-4
                rm, rs = ren[..., k][alpha].mean(), ren[..., k][alpha].std() + 1e-4
                gen[..., k] = np.clip((gen[..., k] - gm) * (rs / gs) + rm, 0, 1)
        mw = np.array(c["matrix"]); w2c = np.linalg.inv(mw)
        pc = pos @ w2c[:3, :3].T + w2c[:3, 3]
        z = -pc[:, 2]; f = c["lens"] / c["sensor"]
        xs = f * pc[:, 0] / np.maximum(z, 1e-6) - c["shift"][0]; ys = f * pc[:, 1] / np.maximum(z, 1e-6) - c["shift"][1]
        u = (xs + 0.5) * res; v = (0.5 - ys) * res
        inside = (z > 0) & (u >= 0) & (u < res - 1) & (v >= 0) & (v < res - 1)
        ui, vi = np.clip(u.astype(int), 0, res - 1), np.clip(v.astype(int), 0, res - 1)
        seen = inside & (rpos[vi, ui, 3] > 0.5) & (np.linalg.norm(rpos[vi, ui, :3] - pos, axis=1) < eps)
        camloc = mw[:3, 3]; view = camloc - pos; dist = np.linalg.norm(view, axis=1); view /= dist[:, None]
        cos = np.clip((nrm * view).sum(1), 0, 1)
        edge = np.clip(np.minimum.reduce([u, v, res - 1 - u, res - 1 - v]) / (0.08 * res), 0, 1)
        w = np.where(seen, cos ** 6 * edge, 0).astype(np.float32)
        # bilinear sample of the generated view, at its own resolution
        ug, vg = u * scale, v * scale
        u0, v0 = np.floor(ug).astype(int).clip(0, gres - 2), np.floor(vg).astype(int).clip(0, gres - 2); fu, fv = (ug - u0)[:, None], (vg - v0)[:, None]
        col = (gen[v0, u0] * (1 - fu) * (1 - fv) + gen[v0, u0 + 1] * fu * (1 - fv) + gen[v0 + 1, u0] * (1 - fu) * fv + gen[v0 + 1, u0 + 1] * fu * fv)
        acc += srgb_to_lin(col) * w[:, None]; wsum += w; best = np.maximum(best, w)
        print("PROJECTED", c["name"], int(seen.sum()), "texels seen")
    out_lin = srgb_to_lin(np.clip(base, 0, 1)).copy()   # Blender holds sRGB images' pixels as sRGB values
    lin = out_lin[valid]
    if blend == "progressive":
        # winner-take-most: a texel moves to this view in proportion to w^k / (w^k + best^k), where best is the strongest
        # view that painted it before; k = 4 gives a narrow feather, k = 16 nearly blits the best view's pixels
        state = f"{out}/best-{size}.npy"; best_prev = np.load(state) if os.path.exists(state) else np.zeros(len(pos), np.float32)
        wv = np.where(wsum > 0, wsum, 0); alpha = (wv ** sharp / (wv ** sharp + best_prev ** sharp + 1e-12))[:, None]
        newc = acc / np.maximum(wsum, 1e-6)[:, None]
        lin = np.where(wsum[:, None] > 1e-4, newc * alpha + lin * (1 - alpha), lin)
        np.save(state, np.maximum(best_prev, wv))
        painted = np.maximum(best_prev, wv) > 1e-4
    else:
        painted = wsum > 1e-4; mix = np.clip(best / 0.05, 0, 1)[:, None]
        lin = np.where(painted[:, None], (acc / np.maximum(wsum, 1e-6)[:, None]) * mix + lin * (1 - mix), lin)
    out_lin[valid] = lin
    tex = lin_to_srgb(out_lin)
    # pad UV islands outward so mip-mapping and filtering never read the empty background
    filled = valid.copy()
    for _ in range(12):
        acc2 = np.zeros_like(tex); n2 = np.zeros(filled.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            acc2 += np.roll(np.roll(tex * filled[..., None], dy, 0), dx, 1); n2 += np.roll(np.roll(filled, dy, 0), dx, 1)
        grow = (~filled) & (n2 > 0); tex[grow] = acc2[grow] / n2[grow][:, None]; filled |= grow
    img = bpy.data.images.new("base_color", size, size, alpha=True)
    img.pixels[:] = np.concatenate([tex, np.ones((size, size, 1), np.float32)], 2).ravel()
    img.filepath_raw = f"{out}/base_color.png"; img.file_format = 'PNG'; img.save()
    # texel density: used texels over surface area (the piece is in metres)
    bm = bmesh.new(); bm.from_mesh(obj.data); area = sum(f.calc_area() for f in bm.faces)
    density = math.sqrt(valid.sum() / area)
    print("COVERAGE %.1f%% of surface texels painted; texel density %.0f px/m at %d^2" % (100 * painted.mean(), density, size))
    use(mat); base_node.image = img; img.pack()
    bpy.ops.wm.save_as_mainfile(filepath=f"{out}/textured.blend", compress=True)
    json.dump(dict(coverage=float(painted.mean()), density=density, tex=size), open(f"{out}/project.json", "w"))
