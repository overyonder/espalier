# Clay render of a raw mesh from four eye-level orthographic views (front, front on the left, back, front on the right)
# framed like the input views: the mesh's height fills FILL of the frame, centred on its bounding box.
# Transparent background. blender -b --python render/render_views.py -- <mesh.glb|mesh.ply> <out prefix> [yaw degrees] [up axis]
# VIEWS="az:el,..." renders those views instead, framing the bounding sphere's diameter in FILL of the frame so tilted
# views aren't cropped (project.py frames an azimuth:elevation view the same way).
# The yaw turns a model whose export faces another way (Pixal3D and TRELLIS.2 face glTF -Z) to the shared front. A PLY
# is imported natively, without glTF's axis conversion, so a Y-up PLY (Hunyuan's) passes up axis y to stand it up.
import bpy, sys, math, mathutils, os
args = sys.argv[sys.argv.index('--') + 1:]; mesh_path, prefix = args[:2]; yaw = float(args[2]) if len(args) > 2 else 0.0
up = args[3] if len(args) > 3 else 'z'
FILL = 0.88
bpy.ops.wm.read_factory_settings(use_empty=True)
if mesh_path.endswith('.ply'): bpy.ops.wm.ply_import(filepath=mesh_path)
else: bpy.ops.import_scene.gltf(filepath=mesh_path)
objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
for o in [o for o in bpy.context.scene.objects if o.parent is None]:
    o.rotation_mode = 'XYZ'; o.rotation_euler.z += math.radians(yaw)
    if up == 'y': o.rotation_euler.x += math.radians(90)
bpy.context.view_layer.update()
mat = bpy.data.materials.new('clay'); mat.use_nodes = True
mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.45, 0.43, 0.41, 1)
if os.environ.get('NORMALS') == '1':   # a camera-space normal pass (x red, y green, toward the camera blue) as emission
    nodes, links = mat.node_tree.nodes, mat.node_tree.links; nodes.remove(nodes['Principled BSDF'])
    geometry, transform = nodes.new('ShaderNodeNewGeometry'), nodes.new('ShaderNodeVectorTransform')
    transform.vector_type, transform.convert_from, transform.convert_to = 'NORMAL', 'WORLD', 'CAMERA'
    scale, emission = nodes.new('ShaderNodeVectorMath'), nodes.new('ShaderNodeEmission')
    scale.operation = 'MULTIPLY_ADD'; scale.inputs[1].default_value = (0.5, 0.5, -0.5); scale.inputs[2].default_value = (0.5, 0.5, 0.5)
    links.new(geometry.outputs['Normal'], transform.inputs['Vector']); links.new(transform.outputs['Vector'], scale.inputs[0])
    links.new(scale.outputs['Vector'], emission.inputs['Color']); links.new(emission.outputs['Emission'], nodes['Material Output'].inputs['Surface'])
if os.environ.get('POSITIONS') == '1':   # a position pass: each point's place in the object's box (0 to 1) as emission
    nodes, links = mat.node_tree.nodes, mat.node_tree.links; nodes.remove(nodes['Principled BSDF'])
    coordinates, emission = nodes.new('ShaderNodeTexCoord'), nodes.new('ShaderNodeEmission')
    links.new(coordinates.outputs['Generated'], emission.inputs['Color']); links.new(emission.outputs['Emission'], nodes['Material Output'].inputs['Surface'])
for o in objs:   # KEEP_MATERIAL=1 renders the mesh's own textured material instead of clay
    if os.environ.get('SMOOTH') == '1':  # match the kit importer for generated GLBs without normals
        for polygon in o.data.polygons: polygon.use_smooth = True
    if os.environ.get('KEEP_MATERIAL') == '1' and o.data.materials: continue
    o.data.materials.clear(); o.data.materials.append(mat)
# ALBEDO_ONLY=1 renders base colour alone: metallic and roughness maps unlinked and fixed at 0 and 0.8, the values the
# painted textures carry, so textures compare by colour under one material.
if os.environ.get('ALBEDO_ONLY') == '1':
    for material in {slot.material for o in objs for slot in o.material_slots if slot.material and slot.material.use_nodes}:
        for node in [n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']:
            for name, value in (('Metallic', 0.0), ('Roughness', 0.8)):
                for link in list(node.inputs[name].links): material.node_tree.links.remove(link)
                node.inputs[name].default_value = value
pts = [o.matrix_world @ mathutils.Vector(c) for o in objs for c in o.bound_box]
lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)]); hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
centre, height, size = (lo + hi) / 2, hi.z - lo.z, max(hi - lo)
# EEVEE, Blender's GPU rasteriser, with its ray-traced screen-space lighting, at 2048 px per view (RENDER_RES
# overrides) so fine relief is visible. RENDER_ENGINE=cycles path-traces with OptiX instead, about ten times slower.
sc = bpy.context.scene
if os.environ.get('RENDER_ENGINE') == 'cycles':
    prefs = bpy.context.preferences.addons['cycles'].preferences; prefs.compute_device_type = 'OPTIX'; prefs.get_devices()
    for device in prefs.devices: device.use = device.type == 'OPTIX'
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'GPU'; sc.cycles.samples = 128; sc.cycles.use_denoising = True
else:
    sc.render.engine = next(e for e in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT') if e in {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items})
    sc.eevee.taa_render_samples = 64; sc.eevee.use_raytracing = True; sc.eevee.use_shadows = True
sc.render.resolution_x = sc.render.resolution_y = int(os.environ.get('RENDER_RES', 2048)); sc.render.film_transparent = True
# Lighting: TRELLIS.2's example environment map for soft fill from every side, a large key light from the upper left
# that moves with the camera, and a rim light behind, with AgX tone mapping so form and small relief both read.
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True; nt = w.node_tree
hdri_path = os.path.expanduser('~/code/models/TRELLIS.2/assets/hdri/forest.exr')
if os.path.exists(hdri_path) and os.environ.get('FLAT') != '1':
    env = nt.nodes.new('ShaderNodeTexEnvironment'); env.image = bpy.data.images.load(hdri_path)
    nt.links.new(env.outputs['Color'], nt.nodes['Background'].inputs['Color']); nt.nodes['Background'].inputs['Strength'].default_value = 0.08
else:
    nt.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    nt.nodes['Background'].inputs['Strength'].default_value = 1.0
passes = os.environ.get('NORMALS') == '1' or os.environ.get('POSITIONS') == '1'   # data passes show their values unmapped
sc.view_settings.view_transform = 'Standard' if passes else 'AgX'
if not passes: sc.view_settings.look = 'AgX - Medium High Contrast'
def area(name, energy, size):
    light = bpy.data.lights.new(name, 'AREA'); light.energy = energy; light.size = size
    obj = bpy.data.objects.new(name, light); sc.collection.objects.link(obj); return obj
key, rim = area('key', 160 * size ** 2, size * 0.6), area('rim', 60 * size ** 2, size * 0.5)
if os.environ.get('FLAT') == '1':   # even white light from every side and no key or rim, for views an image model paints
    nt.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1); nt.nodes['Background'].inputs['Strength'].default_value = 1.0
    if len(nt.nodes['Background'].inputs['Color'].links) > 0:
        nt.links.remove(nt.nodes['Background'].inputs['Color'].links[0])
    key.data.energy = rim.data.energy = 0
    sc.view_settings.look = 'None'
def aim(obj, azimuth, elevation, distance):
    a, e = math.radians(azimuth), math.radians(elevation)
    obj.location = centre + mathutils.Vector((distance * math.cos(e) * math.sin(a), -distance * math.cos(e) * math.cos(a), distance * math.sin(e)))
    obj.rotation_euler = (centre - obj.location).to_track_quat('-Z', 'Y').to_euler()
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
views = [tuple(float(x) for x in v.split(':')) for v in os.environ['VIEWS'].split(',')] if os.environ.get('VIEWS') else [(az, 0.0) for az in (0, 90, 180, 270)]
cam.data.type = 'ORTHO'; cam.data.ortho_scale = ((hi - lo).length if os.environ.get('VIEWS') else height) / FILL; cam.data.clip_end = size * 10
for i, (az, el) in enumerate(views):              # glTF front (+Z) becomes Blender -Y; az 90 puts the front on the image's left
    aim(cam, az, el, size * 3)
    aim(key, az - 40, el + math.copysign(35, el or 1), size * 2.5); aim(rim, az + 150, 25, size * 2.5)   # the key stays on the camera's side of the horizon
    sc.render.filepath = f'{prefix}-{i}.png'; bpy.ops.render.render(write_still=True)
