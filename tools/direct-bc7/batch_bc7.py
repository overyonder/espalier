"""Bounded batch adapter for the direct GPU projection/BC7 shaders.

jobs TSV columns: id, mesh, cameras, paintings (PNG directory), inputs (BC7 cache).
--window sets the number of queued assets. All GPU pipelines are shared. Results
are independent KTX atlases and unlit previews; source models stay untouched.
"""

import os, time, json, resource, threading, ctypes, mmap, struct, statistics
from pathlib import Path

started = time.perf_counter()
cpu_started = resource.getrusage(resource.RUSAGE_SELF)
cpu_started = cpu_started.ru_utime + cpu_started.ru_stime
os.environ["WGPU_BACKEND_TYPE"] = "Vulkan"
import ctypes.util
import numpy as np, wgpu

import csv, argparse
from types import SimpleNamespace

parser = argparse.ArgumentParser(
    description="Batch saved GPT projections through one GPU baker"
)
parser.add_argument("jobs", type=Path, nargs="?")
parser.add_argument("output", type=Path, nargs="?")
parser.add_argument("--window", type=int, default=1)
parser.add_argument(
    "--single", nargs=4, metavar=("MESH", "CAMERAS", "BC7_INPUTS", "OUTPUT")
)
args = parser.parse_args()
if args.single:
    args.output = Path(args.single[3])
    args.window = 1
else:
    assert args.jobs is not None and args.output is not None
assert args.window > 0
shader_dir = Path(__file__).resolve().parent
out = args.output
out.mkdir(parents=True, exist_ok=True)
metrics = {}
stop = threading.Event()
phase = "setup"
util_samples = []
util_timeline = []
phase_vram = {}
phase_ram = {}


class Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class Process(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used", ctypes.c_ulonglong),
        ("gpu", ctypes.c_uint),
        ("compute", ctypes.c_uint),
    ]


nv = ctypes.CDLL("/run/opengl-driver/lib/libnvidia-ml.so.1")
assert nv.nvmlInit_v2() == 0
handle = ctypes.c_void_p()
assert nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0
peak = 0
errors = set()


def sample():
    global peak
    while not stop.is_set():
        for name in [
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses_v2",
        ]:
            count = ctypes.c_uint(64)
            entries = (Process * 64)()
            status = getattr(nv, name)(handle, ctypes.byref(count), entries)
            if status == 0:
                for p in entries[: count.value]:
                    if p.pid == os.getpid() and p.used < 2**63:
                        peak = max(peak, p.used)
                        phase_vram[phase] = max(phase_vram.get(phase, 0), p.used)
            else:
                errors.add(status)
        utilization = Util()
        if nv.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization)) == 0:
            util_samples.append((phase, utilization.gpu))
            util_timeline.append(
                (time.perf_counter() - started, phase, utilization.gpu)
            )
        with open("/proc/self/statm") as status:
            rss = int(status.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        phase_ram[phase] = max(phase_ram.get(phase, 0), rss)
        stop.wait(0.02)


thread = threading.Thread(target=sample, daemon=True)
thread.start()
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync(
    required_limits={"max-compute-workgroup-storage-size": 20480},
    required_features=["texture-compression-bc", "timestamp-query"],
)
queue = device.queue
print("GPU:", adapter.info, flush=True)
metrics["renderer"] = dict(adapter.info)
B = wgpu.BufferUsage
T = wgpu.TextureUsage


def buffer(data, usage=B.STORAGE):
    return device.create_buffer_with_data(data=data, usage=usage | B.COPY_DST)


def bind(pipeline, index, resources):
    return device.create_bind_group(
        layout=pipeline.get_bind_group_layout(index),
        entries=[
            dict(
                binding=k,
                resource=({"buffer": v} if isinstance(v, wgpu.GPUBuffer) else v),
            )
            for k, v in resources.items()
        ],
    )


def compute(pipeline, groups, count):
    encoder = device.create_command_encoder()
    work = encoder.begin_compute_pass()
    work.set_pipeline(pipeline)
    for index, group in enumerate(groups):
        work.set_bind_group(index, group)
    work.dispatch_workgroups(count)
    work.end()
    queue.submit([encoder.finish()])


class PngImage(ctypes.Structure):
    _fields_ = [
        ("opaque", ctypes.c_void_p),
        ("version", ctypes.c_uint),
        ("width", ctypes.c_uint),
        ("height", ctypes.c_uint),
        ("format", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("colormap_entries", ctypes.c_uint),
        ("warning_or_error", ctypes.c_uint),
        ("message", ctypes.c_char * 64),
    ]


png = ctypes.CDLL(
    os.environ.get("LIBPNG", ctypes.util.find_library("png16") or "libpng16.so.16")
)
png.png_image_write_to_file.argtypes = [
    ctypes.POINTER(PngImage),
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]


def write_png(path, buffer, width, height):
    image = PngImage()
    image.version = 1
    image.width = width
    image.height = height
    image.format = 3
    image.flags = 2
    assert png.png_image_write_to_file(
        ctypes.byref(image),
        os.fsencode(path),
        0,
        ctypes.addressof(ctypes.c_char.from_buffer(buffer)),
        width * 4,
        None,
    ), image.message


vertex_source = """
struct Vertex { @builtin(position) clip:vec4<f32>, @location(0) uv:vec2<f32> }
@group(0) @binding(0) var<uniform> vp:mat4x4<f32>;
@vertex fn vertex(@location(0) p:vec3<f32>,@location(1) uv:vec2<f32>)->Vertex {
 let clip=vp*vec4(p.x,-p.z,p.y,1.);var result:Vertex;
 result.clip=vec4(clip.xy,(clip.z+clip.w)*.5,clip.w);result.uv=uv;return result;
}
"""
vertex_layout = [
    dict(
        array_stride=12,
        attributes=[dict(format="float32x3", offset=0, shader_location=0)],
    ),
    dict(
        array_stride=8,
        attributes=[dict(format="float32x2", offset=0, shader_location=1)],
    ),
]
depth_module = device.create_shader_module(code=vertex_source)
depth_pipeline = device.create_render_pipeline(
    layout="auto",
    vertex=dict(module=depth_module, entry_point="vertex", buffers=vertex_layout),
    primitive=dict(topology="triangle-list", cull_mode="none"),
    depth_stencil=dict(
        format="depth32float", depth_write_enabled=True, depth_compare="less"
    ),
)
bin_module = device.create_shader_module(code=(shader_dir / "bin_uv.wgsl").read_text())
bin_pipeline = device.create_compute_pipeline(
    layout="auto", compute=dict(module=bin_module, entry_point="bin_uv")
)
phase = "shader_compilation"
t = time.perf_counter()
print("Compiling fused projection / BC7 shader", flush=True)
source = (
    (shader_dir / "bc7.wgsl")
    .read_text()
    .replace("@group(0) @binding(0) var source_texture: texture_2d<f32>;", "")
    .replace(
        "textureLoad(source_texture, vec2<u32>(pixel_x, pixel_y), 0)",
        "project_pixel(vec2<u32>(pixel_x, pixel_y))",
    )
)
module = device.create_shader_module(
    code=source + "\n" + (shader_dir / "project.wgsl").read_text()
)
pipeline = device.create_compute_pipeline(
    layout="auto", compute=dict(module=module, entry_point="compress_bc7")
)
metrics["shader_compile_seconds"] = time.perf_counter() - t
table_buffer = buffer((shader_dir / "bc7-tables.bin").read_bytes(), B.UNIFORM)
settings = buffer(
    struct.pack("<19I", *[4, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1, 0, 64, 64, 0, 0, 4, 3])
)

sampler = device.create_sampler(
    min_filter="linear",
    mag_filter="linear",
    address_mode_u="clamp-to-edge",
    address_mode_v="clamp-to-edge",
)
preview_source = (
    vertex_source
    + """
@group(0) @binding(1) var atlas:texture_2d<f32>;
@group(0) @binding(2) var linear_sampler:sampler;
@fragment fn fragment(input:Vertex)->@location(0) vec4<f32>{return vec4(textureSample(atlas,linear_sampler,input.uv).rgb,1.);}
"""
)
preview_module = device.create_shader_module(code=preview_source)
preview_pipeline = device.create_render_pipeline(
    layout="auto",
    vertex=dict(module=preview_module, entry_point="vertex", buffers=vertex_layout),
    fragment=dict(
        module=preview_module,
        entry_point="fragment",
        targets=[dict(format="rgba8unorm")],
    ),
    primitive=dict(topology="triangle-list", cull_mode="none"),
    depth_stencil=dict(
        format="depth32float", depth_write_enabled=True, depth_compare="less"
    ),
)


def prepare_asset(job):
    global phase
    item_metrics = {}
    # Mesh accessors are views into the mapped GLB; its old images are never decoded.
    mesh_path = Path(job["mesh"])
    mesh_file = mesh_path.open("rb")
    mapped = mmap.mmap(mesh_file.fileno(), 0, access=mmap.ACCESS_READ)
    json_size = struct.unpack_from("<I", mapped, 12)[0]
    document = json.loads(mapped[20 : 20 + json_size])
    binary_offset = 28 + json_size
    primitive = document["meshes"][0]["primitives"][0]

    def accessor(index):
        a = document["accessors"][index]
        v = document["bufferViews"][a["bufferView"]]
        start = binary_offset + v.get("byteOffset", 0) + a.get("byteOffset", 0)
        length = (
            a["count"]
            * {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
            * {5123: 2, 5125: 4, 5126: 4}[a["componentType"]]
        )
        return memoryview(mapped)[start : start + length]

    position_buffer = buffer(
        accessor(primitive["attributes"]["POSITION"]), B.STORAGE | B.VERTEX
    )
    flat_normals = "NORMAL" not in primitive["attributes"]
    normal_buffer = buffer(
        bytes(4) if flat_normals else accessor(primitive["attributes"]["NORMAL"])
    )
    uv_buffer = buffer(
        accessor(primitive["attributes"]["TEXCOORD_0"]), B.STORAGE | B.VERTEX
    )
    indices = buffer(accessor(primitive["indices"]), B.STORAGE | B.INDEX)
    index_size = {5123: 2, 5125: 4}[
        document["accessors"][primitive["indices"]]["componentType"]
    ]
    vertex_count = document["accessors"][primitive["attributes"]["POSITION"]]["count"]
    face_count = document["accessors"][primitive["indices"]]["count"] // 3

    item_metrics["normal_mode"] = "implicit flat" if flat_normals else "stored vertex"
    camera_data = json.loads(Path(job["cameras"]).read_text())
    cameras = camera_data["cams"]
    assert len(cameras) == 13
    width = height = camera_data["res"]
    padded = (width + 3) // 4 * 4
    paintings = device.create_texture(
        size=(padded, padded, 13),
        format="bc7-rgba-unorm",
        usage=T.COPY_DST | T.TEXTURE_BINDING,
    )
    inputs = Path(job["inputs"])
    t = time.perf_counter()
    uploaded_bytes = 0
    phase = "input_upload"
    for i, c in enumerate(cameras):
        data = (inputs / (c["name"] + ".bc7")).read_bytes()
        uploaded_bytes += len(data)
        queue.write_texture(
            dict(texture=paintings, origin=(0, 0, i)),
            data,
            dict(bytes_per_row=padded * 4),
            (padded, padded, 1),
        )
    del data
    queue.on_submitted_work_done_sync()
    item_metrics["input_upload_seconds"] = time.perf_counter() - t
    item_metrics["input_bc7_bytes"] = uploaded_bytes
    matrices = []
    origins = []
    for c in cameras:
        mw = np.array(c["matrix"], dtype="f4")
        view = np.linalg.inv(mw)
        f = 2 * c["lens"] / c["sensor"]
        near = 0.01
        far = 1000.0
        projection = np.array(
            [
                [f, 0, -2 * c["shift"][0], 0],
                [0, f, -2 * c["shift"][1], 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0],
            ],
            dtype="f4",
        )
        matrices.append(projection @ view)
        origins.append([*mw[:3, 3], 1.0])
    vp_buffer = buffer(matrices[0].T.astype("f4").tobytes(), B.UNIFORM)
    depth_group = bind(depth_pipeline, 0, {0: vp_buffer})
    depths = device.create_texture(
        size=(width, height, 13),
        format="depth32float",
        usage=T.RENDER_ATTACHMENT | T.TEXTURE_BINDING,
    )

    def draw(pipeline, group, attachments, depth_view):
        encoder = device.create_command_encoder()
        work = encoder.begin_render_pass(
            color_attachments=attachments,
            depth_stencil_attachment=dict(
                view=depth_view,
                depth_clear_value=1.0,
                depth_load_op="clear",
                depth_store_op="store",
            ),
        )
        work.set_pipeline(pipeline)
        work.set_bind_group(0, group)
        work.set_vertex_buffer(0, position_buffer)
        work.set_vertex_buffer(1, uv_buffer)
        work.set_index_buffer(indices, "uint16" if index_size == 2 else "uint32")
        work.draw_indexed(face_count * 3)
        work.end()
        queue.submit([encoder.finish()])

    phase = "depth"
    t = time.perf_counter()
    for i, matrix in enumerate(matrices):
        queue.write_buffer(vp_buffer, 0, matrix.T.astype("f4").tobytes())
        draw(
            depth_pipeline,
            depth_group,
            [],
            depths.create_view(dimension="2d", base_array_layer=i, array_layer_count=1),
        )
    queue.on_submitted_work_done_sync()
    item_metrics["depth_seconds"] = time.perf_counter() - t
    phase = "uv_bins"
    t = time.perf_counter()
    print("Building UV triangle bins", flush=True)
    counts = buffer(bytes(128 * 128 * 4), B.STORAGE | B.COPY_SRC)
    offsets_buffer = buffer(bytes((128 * 128 + 1) * 4))
    triangles_buffer = buffer(bytes(4))
    parameters = buffer(struct.pack("<4I", face_count, 0, index_size, 0), B.UNIFORM)
    bin_group = bind(
        bin_pipeline,
        0,
        {
            0: uv_buffer,
            1: indices,
            2: offsets_buffer,
            3: triangles_buffer,
            4: counts,
            5: parameters,
        },
    )
    compute(bin_pipeline, [bin_group], (face_count + 127) // 128)
    bin_counts = np.frombuffer(queue.read_buffer(counts), dtype="u4")
    offsets = np.empty(128 * 128 + 1, dtype="u4")
    offsets[0] = 0
    np.cumsum(bin_counts, out=offsets[1:])
    item_metrics["uv_bin_references"] = int(offsets[-1])
    item_metrics["uv_bins_bytes"] = int(offsets.nbytes + offsets[-1] * 4)
    queue.write_buffer(offsets_buffer, 0, offsets)
    triangles_buffer.destroy()
    triangles_buffer = device.create_buffer(size=int(offsets[-1]) * 4, usage=B.STORAGE)
    queue.write_buffer(counts, 0, bytes(128 * 128 * 4))
    queue.write_buffer(parameters, 0, struct.pack("<4I", face_count, 1, index_size, 0))
    bin_group = bind(
        bin_pipeline,
        0,
        {
            0: uv_buffer,
            1: indices,
            2: offsets_buffer,
            3: triangles_buffer,
            4: counts,
            5: parameters,
        },
    )
    compute(bin_pipeline, [bin_group], (face_count + 127) // 128)
    queue.on_submitted_work_done_sync()
    counts.destroy()
    item_metrics["uv_binning_seconds"] = time.perf_counter() - t
    uniforms = buffer(bytes(16), B.UNIFORM)
    blocks = device.create_buffer(size=1048576, usage=B.STORAGE | B.COPY_SRC)
    camera_bytes = (
        np.asarray(matrices).transpose(0, 2, 1).astype("f4").tobytes()
        + np.asarray(origins, dtype="f4").tobytes()
        + struct.pack("<4I", width, padded, index_size, int(flat_normals))
    )
    camera_buffer = buffer(camera_bytes, B.UNIFORM)
    groups = [
        bind(pipeline, 0, {0: table_buffer, 1: blocks, 2: uniforms, 3: settings}),
        bind(
            pipeline,
            1,
            {
                0: position_buffer,
                1: normal_buffer,
                2: uv_buffer,
                3: indices,
                4: offsets_buffer,
                5: triangles_buffer,
            },
        ),
        bind(
            pipeline,
            2,
            {
                0: paintings.create_view(dimension="2d-array"),
                1: depths.create_view(dimension="2d-array"),
                2: sampler,
                3: camera_buffer,
            },
        ),
    ]
    atlas = device.create_texture(
        size=(4096, 4096, 1),
        format="bc7-rgba-unorm",
        usage=T.COPY_DST | T.TEXTURE_BINDING,
    )
    queries = device.create_query_set(type="timestamp", count=32)
    query_buffer = device.create_buffer(size=256, usage=B.QUERY_RESOLVE | B.COPY_SRC)
    header = b"\xabKTX 11\xbb\r\n\x1a\n" + struct.pack(
        "<13I", 0x04030201, 0, 1, 0, 0x8E8C, 0x1908, 4096, 4096, 0, 0, 1, 1, 0
    )

    return SimpleNamespace(
        job=job,
        metrics=item_metrics,
        mesh_file=mesh_file,
        mapped=mapped,
        positions=position_buffer,
        normals=normal_buffer,
        uv=uv_buffer,
        indices=indices,
        index_size=index_size,
        faces=face_count,
        vertices=vertex_count,
        paintings=paintings,
        depths=depths,
        offsets=offsets_buffer,
        triangles=triangles_buffer,
        parameters=parameters,
        vp=vp_buffer,
        camera=camera_buffer,
        uniforms=uniforms,
        blocks=blocks,
        groups=groups,
        atlas=atlas,
        queries=queries,
        query_readback=device.create_buffer(size=256, usage=B.COPY_DST | B.MAP_READ),
        query_buffer=query_buffer,
        matrices=matrices,
        readbacks=[],
        uniform_buffers=[],
        header=header,
    )


# Input compression uses the same quality profile, without projection.
input_source = (
    (shader_dir / "bc7.wgsl")
    .read_text()
    .replace(
        "@group(0) @binding(0) var source_texture:",
        "@group(1) @binding(0) var source_texture:",
    )
)
input_module = device.create_shader_module(code=input_source)
input_pipeline = device.create_compute_pipeline(
    layout="auto", compute=dict(module=input_module, entry_point="compress_bc7")
)
png.png_image_begin_read_from_file.argtypes = [
    ctypes.POINTER(PngImage),
    ctypes.c_char_p,
]
png.png_image_finish_read.argtypes = [
    ctypes.POINTER(PngImage),
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]


def prepare_inputs(job):
    camera_data = json.loads(Path(job["cameras"]).read_text())
    size = camera_data["res"]
    padded = (size + 3) // 4 * 4
    directory = Path(job["inputs"])
    directory.mkdir(parents=True, exist_ok=True)
    missing = [
        c
        for c in camera_data["cams"]
        if not (directory / (c["name"] + ".bc7")).exists()
    ]
    if not missing:
        return 0
    decoded = bytearray(padded * padded * 4)
    address = ctypes.addressof(ctypes.c_char.from_buffer(decoded))
    pixels = np.frombuffer(decoded, dtype="u1").reshape(padded, padded, 4)
    source = device.create_texture(
        size=(padded, padded, 1),
        format="rgba8unorm",
        usage=T.COPY_DST | T.TEXTURE_BINDING,
    )
    compressed = device.create_buffer(
        size=padded * padded, usage=B.STORAGE | B.COPY_SRC
    )
    uniforms = buffer(struct.pack("<4I", padded, padded, 0, 0), B.UNIFORM)
    groups = [
        bind(
            input_pipeline,
            0,
            {0: table_buffer, 1: compressed, 2: uniforms, 3: settings},
        ),
        bind(input_pipeline, 1, {0: source.create_view()}),
    ]
    for camera in missing:
        image = PngImage()
        image.version = 1
        assert png.png_image_begin_read_from_file(
            ctypes.byref(image),
            os.fsencode(Path(job["paintings"]) / (camera["name"] + ".png")),
        ), image.message
        assert (image.width, image.height) == (size, size)
        image.format = 3
        image.flags |= 4
        assert png.png_image_finish_read(
            ctypes.byref(image), None, address, -padded * 4, None
        ), image.message
        pixels[:size, size:] = pixels[:size, size - 1 : size]
        pixels[size:] = pixels[size - 1 : size]
        queue.write_texture(
            dict(texture=source),
            decoded,
            dict(bytes_per_row=padded * 4),
            (padded, padded, 1),
        )
        encoder = device.create_command_encoder()
        work = encoder.begin_compute_pass()
        work.set_pipeline(input_pipeline)
        for index, group in enumerate(groups):
            work.set_bind_group(index, group)
        work.dispatch_workgroups((padded // 4 + 31) // 32, padded // 4, 1)
        work.end()
        queue.submit([encoder.finish()])
        (directory / (camera["name"] + ".bc7")).write_bytes(
            queue.read_buffer(compressed)
        )
    source.destroy()
    compressed.destroy()
    uniforms.destroy()
    return len(missing)


def submit_asset(asset):
    encoder = device.create_command_encoder()
    for strip, row in enumerate(range(0, 4096, 256)):
        uniforms = buffer(struct.pack("<4I", 4096, 256, row, 0), B.UNIFORM)
        asset.uniform_buffers.append(uniforms)
        group = bind(
            pipeline, 0, {0: table_buffer, 1: asset.blocks, 2: uniforms, 3: settings}
        )
        work = encoder.begin_compute_pass(
            timestamp_writes=dict(
                query_set=asset.queries,
                beginning_of_pass_write_index=strip * 2,
                end_of_pass_write_index=strip * 2 + 1,
            )
        )
        work.set_pipeline(pipeline)
        work.set_bind_group(0, group)
        work.set_bind_group(1, asset.groups[1])
        work.set_bind_group(2, asset.groups[2])
        work.dispatch_workgroups(32, 64, 1)
        work.end()
        encoder.copy_buffer_to_texture(
            dict(buffer=asset.blocks, offset=0, bytes_per_row=16384),
            dict(texture=asset.atlas, origin=(0, row, 0)),
            (4096, 256, 1),
        )
        readback = device.create_buffer(size=1048576, usage=B.COPY_DST | B.MAP_READ)
        asset.readbacks.append(readback)
        encoder.copy_buffer_to_buffer(asset.blocks, 0, readback, 0, 1048576)
    encoder.resolve_query_set(asset.queries, 0, 32, asset.query_buffer, 0)
    encoder.copy_buffer_to_buffer(asset.query_buffer, 0, asset.query_readback, 0, 256)
    asset.submitted = time.perf_counter()
    queue.submit([encoder.finish()])


def finish_asset(asset):
    destination = out if args.single else out / asset.job["id"]
    destination.mkdir(exist_ok=True)
    with (destination / "atlas-bc7.ktx").open("wb") as f:
        f.write(asset.header)
        f.write(struct.pack("<I", 16777216))
        for readback in asset.readbacks:
            readback.map_sync(wgpu.MapMode.READ)
            view = readback.read_mapped(copy=False)
            f.write(view)
            del view
            readback.unmap()
            readback.destroy()
    asset.query_readback.map_sync(wgpu.MapMode.READ)
    timestamps = np.frombuffer(asset.query_readback.read_mapped(), dtype="u8").copy()
    asset.query_readback.unmap()
    asset.metrics["gpu_bake_seconds"] = (
        float(np.sum(timestamps[1::2] - timestamps[::2])) / 1e9
    )
    asset.metrics["submission_to_saved_seconds"] = time.perf_counter() - asset.submitted
    asset.metrics.update(vertices=asset.vertices, faces=asset.faces)
    (destination / "timings.json").write_text(json.dumps(asset.metrics, indent=2))
    for name in (
        "paintings",
        "depths",
        "offsets",
        "triangles",
        "parameters",
        "camera",
        "uniforms",
        "blocks",
        "queries",
        "query_buffer",
        "query_readback",
    ):
        getattr(asset, name).destroy()
    for uniforms in asset.uniform_buffers:
        uniforms.destroy()
    asset.uniform_buffers.clear()
    asset.readbacks.clear()
    asset.groups.clear()
    print("BAKED", asset.job["id"], asset.metrics["gpu_bake_seconds"], flush=True)
    return asset.metrics


def render_asset(asset):
    destination = out if args.single else out / asset.job["id"]
    queue.write_buffer(asset.vp, 0, asset.matrices[0].T.astype("f4").tobytes())
    group = bind(
        preview_pipeline, 0, {0: asset.vp, 1: asset.atlas.create_view(), 2: sampler}
    )
    preview = device.create_texture(
        size=(1080, 1080, 1),
        format="rgba8unorm",
        usage=T.RENDER_ATTACHMENT | T.COPY_SRC,
    )
    depth = device.create_texture(
        size=(1080, 1080, 1), format="depth32float", usage=T.RENDER_ATTACHMENT
    )
    encoder = device.create_command_encoder()
    work = encoder.begin_render_pass(
        color_attachments=[
            dict(
                view=preview.create_view(),
                clear_value=(0.21, 0.21, 0.21, 1.0),
                load_op="clear",
                store_op="store",
            )
        ],
        depth_stencil_attachment=dict(
            view=depth.create_view(),
            depth_clear_value=1.0,
            depth_load_op="clear",
            depth_store_op="store",
        ),
    )
    work.set_pipeline(preview_pipeline)
    work.set_bind_group(0, group)
    work.set_vertex_buffer(0, asset.positions)
    work.set_vertex_buffer(1, asset.uv)
    work.set_index_buffer(
        asset.indices, "uint16" if asset.index_size == 2 else "uint32"
    )
    work.draw_indexed(asset.faces * 3)
    work.end()
    queue.submit([encoder.finish()])
    pixels = bytearray(
        queue.read_texture(
            dict(texture=preview), dict(bytes_per_row=4320), (1080, 1080, 1)
        )
    )
    write_png(destination / "same-camera.png", pixels, 1080, 1080)
    asset.metrics.update(vertices=asset.vertices, faces=asset.faces)
    (destination / "timings.json").write_text(json.dumps(asset.metrics, indent=2))
    preview.destroy()
    depth.destroy()
    for name in ("positions", "normals", "uv", "indices", "vp", "atlas"):
        getattr(asset, name).destroy()
    for uniforms in asset.uniform_buffers:
        uniforms.destroy()
    asset.mapped.close()
    asset.mesh_file.close()
    asset.metrics["saved_id"] = asset.job["id"]
    print("FINISHED", asset.job["id"], asset.metrics["gpu_bake_seconds"], flush=True)
    return asset.metrics


if args.single:
    mesh, cameras, inputs, _ = args.single
    jobs = [
        dict(
            id=Path(mesh).parent.parent.name,
            mesh=mesh,
            cameras=cameras,
            inputs=inputs,
            paintings="",
        )
    ]
else:
    with args.jobs.open() as f:
        jobs = list(csv.DictReader(f, delimiter="\t"))
assert jobs and len({j["id"] for j in jobs}) == len(jobs)
phase = "input_compression"
t = time.perf_counter()
metrics["newly_encoded_inputs"] = sum(prepare_inputs(job) for job in jobs)
metrics["input_compression_seconds"] = time.perf_counter() - t
phase = "batch"
t = time.perf_counter()
results = []
preview_seconds = 0.0
for start in range(0, len(jobs), args.window):
    assets = [prepare_asset(job) for job in jobs[start : start + args.window]]
    # Queue all strips without a CPU readback barrier between dispatches.
    phase = "queued_bakes"
    submitted = time.perf_counter()
    interval_started = submitted - started
    for asset in assets:
        submit_asset(asset)
    phase = "queued_bakes"
    for asset in assets:
        results.append(finish_asset(asset))
    metrics.setdefault("queued_interval_starts", []).append(interval_started)
    metrics.setdefault("queued_bake_intervals", []).append(
        time.perf_counter() - submitted
    )
    phase = "previews"
    preview_started = time.perf_counter()
    for asset in assets:
        render_asset(asset)
    preview_seconds += time.perf_counter() - preview_started
    assets.clear()
metrics["batch_seconds"] = time.perf_counter() - t
metrics["previews_seconds"] = preview_seconds
metrics["gpu_bake_seconds"] = sum(r["gpu_bake_seconds"] for r in results)
metrics["items"] = results
metrics["window"] = args.window
metrics["total_seconds"] = time.perf_counter() - started
cpu = resource.getrusage(resource.RUSAGE_SELF)
metrics["cpu_seconds"] = cpu.ru_utime + cpu.ru_stime - cpu_started
metrics["peak_ram_mib"] = cpu.ru_maxrss / 1024
stop.set()
thread.join()
metrics["peak_process_vram_mib"] = peak / 1048576
metrics["gpu_device_utilization"] = {
    p: dict(
        mean=statistics.mean(v for q, v in util_samples if q == p),
        peak=max(v for q, v in util_samples if q == p),
        samples=sum(q == p for q, v in util_samples),
    )
    for p in set(q for q, v in util_samples)
}
metrics["phase_peak_vram_mib"] = {k: v / 1048576 for k, v in phase_vram.items()}
metrics["phase_peak_rss_mib"] = {k: v / 1048576 for k, v in phase_ram.items()}
metrics["uncompressed_atlas_bytes"] = 0
metrics["atlas_bytes_per_item"] = 16777216
metrics["encoder_workgroup_scratch_bytes"] = 20480
metrics["gpu_dispatch_fraction_of_queued_intervals"] = metrics[
    "gpu_bake_seconds"
] / sum(metrics["queued_bake_intervals"])
metrics["gpu_utilization_timeline"] = util_timeline
metrics["gpu_utilization_scope"] = "NVML whole-device samples, not shader occupancy"
if args.single:
    metrics.update(results[0])
    metrics["projection_and_encoding_gpu_seconds"] = results[0]["gpu_bake_seconds"]
    metrics["projection_and_encoding_wall_seconds"] = metrics["queued_bake_intervals"][
        0
    ]
(out / "timings.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2), flush=True)
