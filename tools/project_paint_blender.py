"""Progressive projection painting's Blender side (albedo_progressive.py drives it).

  blender -b --python project_paint_blender.py -- piece   <base.glb> <piece.blend>
  blender -b --python project_paint_blender.py -- render  <piece.blend> <dir> [views=orbit] [tiles=1] [res=1024]
  (paint <dir>/views/<name>.png -> <dir>/gen/<name>.png with paint_views.py)
  blender -b --python project_paint_blender.py -- project <piece.blend> <dir> [tex=2048] [only=<view>] [blend=progressive]
  blender -b --python project_paint_blender.py -- export  <textured.blend> <painted.glb>

render: cameras orbit the piece (8 azimuths at 15 degrees elevation, 4 at 50, one from above), each optionally split into
tiles x tiles sensor-shift cameras with the lens scaled up, so a tile is the same perspective at tiles-times the pixel
density. Each camera writes the current base colour lit flat (views/<name>.png) and the surface position per pixel
(pos/<name>.exr); cams.json holds the cameras.
project: uploads cached surface maps and the current atlas into CUDA, then applies the selected view with visibility,
facing/edge weighting, progressive blending and UV padding. With no selection, applies the saved views in camera order.
serve: the progressive driver keeps this Blender process and its CUDA atlas alive across render/project/export commands.
Each projection writes base_color.png, best-<size>.npy, project.json and step/textured.blend for resuming.
The overlap exponent is sharp=4 by default. freeze=<atlas mask.npy> protects previously painted surface texels.
Views render in EEVEE; UV surface position and normal bakes run in Cycles once and are cached in <dir>.
Run 6-albedo/build.sh once before direct Blender calls; albedo_progressive.py builds before starting the worker.
"""
import bpy, bmesh, sys, os, json, math, ctypes
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
    bpy.ops.wm.open_mainfile(filepath=src)
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for polygon in obj.data.polygons: polygon.use_smooth = True
    bpy.ops.export_scene.gltf(filepath=out, export_format='GLB'); sys.exit(0)
os.makedirs(out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
if os.path.realpath(src) == os.path.realpath(f"{out}/step/textured.blend"):
    # Open the resume file as the main document so Blender permits saving back to the same path.
    bpy.ops.wm.open_mainfile(filepath=src)
    obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
    sc = bpy.context.scene
else:
    with bpy.data.libraries.load(src) as (f, t): t.objects = f.objects
    obj = next(o for o in t.objects if o.type == 'MESH')
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

def use(m):
    obj.data.materials.clear(); obj.data.materials.append(m)

def cameras(kind, tiles):
    views = (
        [(az, 15) for az in range(0, 360, 45)]
        + [(az, 50) for az in (45, 135, 225, 315)]
        + [(az, -50) for az in (0, 120, 240)]
    )
    fov = math.radians(30); dist = radius / math.sin(fov / 2) * 1.05
    cams = []
    for az, el in views:
        d = Vector((math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(az)) * math.cos(math.radians(el)),
                    math.sin(math.radians(el))))
        loc = centre + d * dist
        rot = (centre - loc).to_track_quat('-Z', 'Y').to_matrix().to_4x4()
        mw = Matrix.Translation(loc) @ rot
        # aim at the silhouette's centre and fit the lens to it, with 4% margin each side
        w2c = np.array(mw.inverted()); pc = pts @ w2c[:3, :3].T + w2c[:3, 3]; sx_, sy_ = pc[:, 0] / -pc[:, 2], pc[:, 1] / -pc[:, 2]
        cx_, cy_ = (sx_.max() + sx_.min()) / 2, (sy_.max() + sy_.min()) / 2
        half = max(sx_.max() - sx_.min(), sy_.max() - sy_.min()) / 2 / 0.92
        aim = (rot.to_3x3() @ Vector((cx_, cy_, -1))).normalized()
        mw = Matrix.Translation(loc) @ aim.to_track_quat('-Z', 'Y').to_matrix().to_4x4()
        lens = float(36 / (2 * half))
        cam_prefix = f"a{az:03d}e{el:02d}" if el >= 0 else f"a{az:03d}em{abs(el):02d}"
        for ty in range(tiles):
            for tx in range(tiles):
                # a tile: the same camera with the lens scaled by `tiles` and the sensor shifted onto that tile
                sx = (tx + 0.5) / tiles - 0.5; sy = 0.5 - (ty + 0.5) / tiles
                cams.append(dict(name=cam_prefix + (f"t{ty}{tx}" if tiles > 1 else ""), matrix=[list(r) for r in mw],
                                 lens=lens * tiles, sensor=36.0, shift=[sx * tiles, sy * tiles]))
    return cams

def make_cam(c, res):
    cd = bpy.data.cameras.new(c["name"]); cd.lens = c["lens"]; cd.sensor_width = c["sensor"]; cd.sensor_fit = 'HORIZONTAL'
    cd.shift_x, cd.shift_y = c["shift"]; cd.clip_start = 0.01; cd.clip_end = 1000
    co = bpy.data.objects.new(c["name"], cd); sc.collection.objects.link(co); co.matrix_world = Matrix(c["matrix"])
    sc.camera = co; sc.render.resolution_x = sc.render.resolution_y = res
    return co

def render_camera(c, res, cached_position=False):
    make_cam(c, res)
    for sub in ("views", "pos"): os.makedirs(f"{out}/{sub}", exist_ok=True)
    sc.render.engine = 'BLENDER_EEVEE'
    use(M_COL); sc.eevee.taa_render_samples = 16; sc.render.filter_size = 1.5
    sc.view_settings.view_transform = 'Standard'
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGBA'
    sc.render.film_transparent = True; sc.render.filepath = f"{out}/views/{c['name']}.png"
    bpy.ops.render.render(write_still=True)
    if not cached_position or not os.path.exists(f"{out}/pos/{c['name']}.exr"):
        use(M_POS); sc.eevee.taa_render_samples = 1; sc.render.filter_size = 0.0; sc.view_settings.view_transform = 'Raw'
        sc.render.image_settings.file_format = 'OPEN_EXR'; sc.render.image_settings.color_depth = '32'
        sc.render.image_settings.color_mode = 'RGBA'; sc.render.filepath = f"{out}/pos/{c['name']}.exr"
        bpy.ops.render.render(write_still=True)
    use(mat)

if mode == "render":
    tiles, res = int(opt.get("tiles", 1)), int(opt.get("res", 1024))
    only = opt.get("only")
    if only and os.path.exists(f"{out}/cams.json"):
        meta = json.load(open(f"{out}/cams.json")); res = meta["res"]
        cams = [c for c in meta["cams"] if c["name"] == only]
    else: cams = cameras(opt.get("views", "orbit"), tiles)
    for c in cams: render_camera(c, res)
    if not only: json.dump(dict(res=res, tiles=tiles, cams=cams), open(f"{out}/cams.json", "w"), indent=1)
    print("RENDERED", len(cams), "views")

def bake(m, size, name):
    img = bpy.data.images.new(name, size, size, float_buffer=True, alpha=True); img.generated_color = (0, 0, 0, 0)
    use(m); n = m.node_tree.nodes.new("ShaderNodeTexImage"); n.image = img; m.node_tree.nodes.active = n
    sc.render.engine = 'CYCLES'; sc.cycles.samples = 1; sc.render.bake.margin = 0
    bpy.context.view_layer.objects.active = obj; obj.select_set(True)
    bpy.ops.object.bake(type='EMIT')
    result = pixels(img).reshape(size, size, 4)
    m.node_tree.nodes.remove(n); bpy.data.images.remove(img)
    if not np.any(result[..., :3]):
        raise RuntimeError(f"Cycles returned an empty {name} surface bake; inspect its render errors")
    return result

def pixels(image):
    data = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(data)
    return data

def load(path):
    i = bpy.data.images.load(path); i.colorspace_settings.name = 'Non-Color' if path.endswith(".exr") else 'sRGB'
    w, h = i.size; p = pixels(i).reshape(h, w, i.channels)
    bpy.data.images.remove(i); return p[::-1]   # top row first

if mode in ("project", "serve"):
    import time
    size = int(opt.get("tex", 4096))
    meta_path = f"{out}/cams.json"
    if not os.path.exists(meta_path):
        json.dump(dict(res=int(opt.get("res", 1254)), tiles=1, cams=cameras("orbit", 1)), open(meta_path, "w"))
    meta = json.load(open(meta_path)); res = meta["res"]
    cache = f"{out}/surface-{size}.npz"
    if os.path.exists(cache):
        with np.load(cache) as surface: P, Nm = surface["P"], surface["N"]
    else:
        P = bake(M_POS, size, "P"); Nm = bake(M_NRM, size, "N"); np.savez(cache, P=P, N=Nm)
    # This mask orders the saved compact weight vector; surface preparation and projection run in CUDA.
    valid = np.einsum('ijk,ijk->ij', Nm[..., :3], Nm[..., :3]) > .01
    if not valid.any(): raise RuntimeError("Cached surface bake contains no UV texels")
    frozen = np.load(opt["freeze"]).astype(np.uint8) if "freeze" in opt else np.zeros((size, size), np.uint8)
    if frozen.shape != valid.shape: raise ValueError("Frozen mask does not match the texture atlas")
    best = np.zeros((size, size), np.float32); state = f"{out}/best-{size}.npy"
    if os.path.exists(state): best[valid] = np.load(state)
    if tuple(base_img.size) != (size, size): base_img.scale(size, size)
    rgba = pixels(base_img).reshape(size, size, 4)
    so_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Intermediate/project_gpu.so"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Intermediate/project_gpu.so"),
    ]
    so_path = next((p for p in so_candidates if os.path.isfile(p)), None)
    if not so_path: raise FileNotFoundError("project_gpu.so not found")
    native = ctypes.CDLL(so_path)
    pointer = ctypes.c_void_p
    native.projection_error.restype = ctypes.c_char_p
    native.projection_create.argtypes = [ctypes.c_uint, pointer, pointer, pointer, pointer, pointer] + [ctypes.c_double] * 4
    native.projection_create.restype = pointer
    native.projection_paint.argtypes = [pointer, pointer, pointer, ctypes.c_uint, pointer, ctypes.c_uint]
    native.projection_download.argtypes = [pointer, pointer, pointer, pointer]
    native.projection_destroy.argtypes = [pointer]
    projector = native.projection_create(size, P.ctypes.data, Nm.ctypes.data, rgba.ctypes.data, best.ctypes.data,
                                         frozen.ctypes.data, PS, *PSHIFT)
    if not projector: raise RuntimeError(native.projection_error().decode())
    del P, Nm, frozen
    counts = np.zeros(3, np.uint32)
    bm = bmesh.new(); bm.from_mesh(obj.data); area = sum(f.calc_area() for f in bm.faces); bm.free()
    density = math.sqrt(valid.sum() / area)
    os.makedirs(f"{out}/step", exist_ok=True)

    def project_camera(name):
        global base_img
        c = next(c for c in meta["cams"] if c["name"] == name)
        gen = np.ascontiguousarray(load(f"{out}/gen/{name}.png"))
        rpos = np.ascontiguousarray(load(f"{out}/pos/{name}.exr"))
        if gen.shape[0] != gen.shape[1] or gen.shape[2] != 4: raise ValueError("Painting must be square RGBA")
        camera = np.array([*np.linalg.inv(np.array(c["matrix"], dtype=np.float64))[:3].ravel(),
                           *np.array(c["matrix"])[:3, 3], c["lens"] / c["sensor"], *c["shift"], res,
                           .004 * radius * 2, PS, *PSHIFT, float(opt.get("sharp", 4))], np.float64)
        started = time.perf_counter()
        if native.projection_paint(projector, camera.ctypes.data, gen.ctypes.data, gen.shape[0], rpos.ctypes.data, res):
            raise RuntimeError(native.projection_error().decode())
        elapsed = time.perf_counter() - started
        if native.projection_download(projector, rgba.ctypes.data, best.ctypes.data, counts.ctypes.data):
            raise RuntimeError(native.projection_error().decode())
        # A fresh datablock gives both .blend checkpoints and glTF export the current packed PNG.
        # Repacking an already packed source image can retain its original compressed bytes.
        previous_image = base_img
        base_img = bpy.data.images.new("base_color", size, size, alpha=True)
        base_img.pixels.foreach_set(rgba.ravel()); base_img.update()
        base_img.filepath_raw = f"{out}/base_color.png"; base_img.file_format = 'PNG'; base_img.save(); base_img.pack()
        base_node.image = base_img
        for node in M_COL.node_tree.nodes:
            if node.type == 'TEX_IMAGE': node.image = base_img
        if previous_image.users == 0: bpy.data.images.remove(previous_image)
        np.save(state, best[valid])
        use(mat); base_node.image = base_img
        bpy.ops.wm.save_as_mainfile(filepath=f"{out}/step/textured.blend", compress=True)
        coverage = float(counts[2]) / max(1, int(counts[1]))
        json.dump(dict(coverage=coverage, density=density, tex=size), open(f"{out}/project.json", "w"))
        print(f"GPU_PROJECT {name} {elapsed:.4f}s; checkpoint {time.perf_counter()-started:.3f}s; "
              f"{counts[0]} texels seen; coverage {coverage:.1%}", flush=True)

    try:
        if mode == "project":
            if opt.get("blend", "progressive") != "progressive": raise ValueError("Use progressive projection blending")
            for c in meta["cams"]:
                if not opt.get("only") or c["name"] == opt["only"]: project_camera(c["name"])
        else:
            def reply(value): print("PROJECTOR " + json.dumps(value), flush=True)
            reply(dict(ready=True, cams=[c["name"] for c in meta["cams"]]))
            for command in sys.stdin:
                request = json.loads(command)
                if request["op"] == "render":
                    c = next(c for c in meta["cams"] if c["name"] == request["name"])
                    render_camera(c, res, cached_position=True)
                elif request["op"] == "project": project_camera(request["name"])
                elif request["op"] == "export":
                    use(mat)
                    for polygon in obj.data.polygons: polygon.use_smooth = True
                    bpy.ops.export_scene.gltf(filepath=request["path"], export_format='GLB')
                elif request["op"] == "quit":
                    reply(dict(done=True)); break
                else: raise ValueError("Unknown projector command")
                reply(dict(done=True))
    finally: native.projection_destroy(projector)
