"""Thin CUDA tensor/Vulkan driver adapter for the resident bake."""

import ctypes as C
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import nvdiffrast.torch as dr
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
SHADERS = ROOT.parent / "direct-bc7"


class Binding(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("group", "binding", "kind", "resource")]


class Vulkan:
    def __init__(self):
        so_candidates = [
            Path(os.environ.get("RESIDENT_VULKAN_LIBRARY", "")),
            ROOT.parent / "Intermediate/resident-depth/resident_vk.so",
            ROOT.parent.parent / "Intermediate/resident-depth/resident_vk.so",
        ]
        so_path = next((p for p in so_candidates if p.is_file()), ROOT.parent / "Intermediate/resident-depth/resident_vk.so")
        self.lib = C.CDLL(str(so_path))
        signatures = {
            "rd_create": (C.c_void_p, []),
            "rd_error": (C.c_char_p, []),
            "rd_destroy": (None, [C.c_void_p]),
            "rd_buffer": (C.c_int, [C.c_void_p, C.c_size_t]),
            "rd_write_mesh": (
                C.c_int,
                [
                    C.c_void_p,
                    C.c_uint32,
                    C.c_uint32,
                    C.c_uint32,
                    C.c_uint32,
                    C.c_char_p,
                ],
            ),
            "rd_write_atlas": (C.c_int, [C.c_void_p, C.c_uint32, C.c_char_p]),
            "rd_write": (
                C.c_int,
                [C.c_void_p, C.c_uint32, C.c_uint64, C.c_size_t, C.c_int],
            ),
            "rd_read": (C.c_int, [C.c_void_p, C.c_uint32, C.c_void_p, C.c_size_t]),
            "rd_read_at": (
                C.c_int,
                [C.c_void_p, C.c_uint32, C.c_size_t, C.c_void_p, C.c_size_t],
            ),
            "rd_clear": (C.c_int, [C.c_void_p, C.c_uint32]),
            "rd_image": (
                C.c_int,
                [C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32],
            ),
            "rd_copy_image": (
                C.c_int,
                [C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32],
            ),
            "rd_pipeline": (
                C.c_int,
                [C.c_void_p, C.c_char_p, C.POINTER(Binding), C.c_uint32, C.c_char_p],
            ),
            "rd_dispatch": (C.c_int, [C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32]),
        }
        for name, (result, args) in signatures.items():
            fn = getattr(self.lib, name)
            fn.restype = result
            fn.argtypes = args
        self.context = self.lib.rd_create()
        if not self.context:
            raise RuntimeError(self.lib.rd_error().decode())

    def call(self, name, *args):
        result = getattr(self.lib, name)(self.context, *args)
        if result < 0:
            raise RuntimeError(self.lib.rd_error().decode())
        return result

    def buffer(self, data=None, size=None):
        if data is not None:
            size = (
                data.numel() * data.element_size()
                if isinstance(data, torch.Tensor)
                else len(data)
            )
        index = self.call("rd_buffer", size)
        if data is not None:
            self.write(index, data)
        return index

    def write(self, index, data):
        if isinstance(data, torch.Tensor):
            assert data.is_cuda and data.is_contiguous()
            torch.cuda.synchronize()
            self.call(
                "rd_write",
                index,
                data.data_ptr(),
                data.numel() * data.element_size(),
                1,
            )
        else:
            self.call(
                "rd_write",
                index,
                C.addressof(data)
                if isinstance(data, C.Array)
                else C.cast(C.c_char_p(data), C.c_void_p).value,
                len(data),
                0,
            )

    def read(self, index, size):
        target = C.create_string_buffer(size)
        self.call("rd_read", index, target, size)
        return target.raw

    def pipeline(self, name, bindings, entry="compress_bc7"):
        array = (Binding * len(bindings))(*(Binding(*b) for b in bindings))
        shader_candidates = [
            Path(os.environ.get("RESIDENT_SHADER_DIRECTORY", "")),
            ROOT.parent / "Intermediate/resident-depth",
            ROOT.parent.parent / "Intermediate/resident-depth",
        ]
        shader_dir = next((p for p in shader_candidates if p.is_dir()), ROOT.parent / "Intermediate/resident-depth")
        return self.call(
            "rd_pipeline",
            str(shader_dir / (name + ".spv")).encode(),
            array,
            len(bindings),
            entry.encode(),
        )

    def close(self):
        self.lib.rd_destroy(self.context)
        self.context = None


def bake(
    unwrapped,
    matrices,
    camera_data,
    paint_paths,
    output_directory,
    context,
    flat_normals=False,
):
    from resident import render_depths

    gpu = Vulkan()
    timings = {}
    started = time.perf_counter()
    try:
        count = len(camera_data["cams"])
        if not 1 <= count <= 16 or len(paint_paths) != count or len(matrices) != count:
            raise ValueError("Matching camera transforms and paintings required (1–16)")
        source_sizes = []
        for path in paint_paths:
            with Image.open(path) as image:
                if image.width != image.height:
                    raise ValueError("Paintings must preserve the square camera canvas")
                source_sizes.append(image.width)
        width = (max(source_sizes) + 3) // 4 * 4
        rgba_image = gpu.call("rd_image", width, width, 1, 37)  # R8G8B8A8_UNORM
        paint_image = gpu.call("rd_image", width, width, count, 145)  # BC7_UNORM_BLOCK
        image_upload = gpu.buffer(size=width * width * 4)
        encoded_input = gpu.buffer(size=width * width)
        tables = gpu.buffer((SHADERS / "bc7-tables.bin").read_bytes())
        settings = gpu.buffer(
            struct.pack(
                "<19I", 4, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1, 0, 64, 64, 0, 0, 4, 3
            )
        )
        uniforms = gpu.buffer(struct.pack("<4I", width, width, 0, 0))
        encoder = gpu.pipeline(
            "encode",
            [
                (0, 0, 0, tables),
                (0, 1, 1, encoded_input),
                (0, 2, 0, uniforms),
                (0, 3, 1, settings),
                (1, 0, 2, rgba_image),
            ],
        )
        stage = time.perf_counter()
        from png_io import Decoder

        decoder = Decoder(width)
        for layer, path in enumerate(paint_paths):
            gpu.write(image_upload, decoder.decode(path))
            gpu.call("rd_copy_image", image_upload, rgba_image, 0)
            gpu.call("rd_dispatch", encoder, (width // 4 + 31) // 32, width // 4)
            gpu.call("rd_copy_image", encoded_input, paint_image, layer)
        timings["encode_inputs_seconds"] = time.perf_counter() - stage
        positions, normals, uv, indices = unwrapped.result()
        stage = time.perf_counter()
        mesh_buffers = [
            gpu.buffer(t.contiguous()) for t in (positions, normals, uv, indices)
        ]
        offsets = gpu.buffer(size=16385 * 4)
        counts = gpu.buffer(size=16384 * 4)
        dummy = gpu.buffer(size=4)
        parameters = gpu.buffer(struct.pack("<4I", len(indices), 0, 4, 0))
        gpu.call("rd_clear", counts)
        counting = gpu.pipeline(
            "bin",
            [
                (0, 0, 1, mesh_buffers[2]),
                (0, 1, 1, mesh_buffers[3]),
                (0, 2, 1, offsets),
                (0, 3, 1, dummy),
                (0, 4, 1, counts),
                (0, 5, 0, parameters),
            ],
            "bin_uv",
        )
        gpu.call("rd_dispatch", counting, (len(indices) + 127) // 128, 1)
        prefix = gpu.pipeline(
            "prefix", [(0, 0, 1, counts), (0, 1, 1, offsets)], "prefix"
        )
        gpu.call("rd_dispatch", prefix, 1, 1)
        total = C.c_uint32()
        gpu.call("rd_read_at", offsets, 16384 * 4, C.byref(total), 4)
        triangles = gpu.buffer(size=max(4, total.value * 4))
        gpu.call("rd_clear", counts)
        gpu.write(parameters, struct.pack("<4I", len(indices), 1, 4, 0))
        binning = gpu.pipeline(
            "bin",
            [
                (0, 0, 1, mesh_buffers[2]),
                (0, 1, 1, mesh_buffers[3]),
                (0, 2, 1, offsets),
                (0, 3, 1, triangles),
                (0, 4, 1, counts),
                (0, 5, 0, parameters),
            ],
            "bin_uv",
        )
        gpu.call("rd_dispatch", binning, (len(indices) + 127) // 128, 1)
        mesh_buffers.extend((offsets, triangles))
        timings["gpu_uv_index_seconds"] = time.perf_counter() - stage
        # Sparse visibility corresponds to the surface receiving the projection.
        depth = (
            render_depths(positions, indices, matrices, camera_data["res"], context)
            .flip(1)
            .contiguous()
        )
        resolution = camera_data["res"]
        depth_image = gpu.call(
            "rd_image", resolution, resolution, count, 100
        )  # R32_SFLOAT
        depth_upload = gpu.buffer(size=resolution * resolution * 4)
        for layer in range(count):
            gpu.write(depth_upload, depth[layer])
            gpu.call("rd_copy_image", depth_upload, depth_image, layer)
        del depth
        origins = np.asarray(
            [
                list(np.asarray(c["matrix"])[:3, 3]) + [source_sizes[i] / width]
                for i, c in enumerate(camera_data["cams"])
            ],
            dtype="<f4",
        )
        padded_matrices = np.zeros((16, 4, 4), dtype="<f4")
        padded_matrices[:count] = matrices.transpose(1, 2).cpu().numpy()
        padded_origins = np.zeros((16, 4), dtype="<f4")
        padded_origins[:count] = origins
        camera_bytes = (
            padded_matrices.tobytes()
            + padded_origins.tobytes()
            + struct.pack("<4I", width, width, 4, int(flat_normals))
        )
        cameras = gpu.buffer(camera_bytes)
        padded_inverse = np.zeros((16, 4, 4), dtype="<f4")
        padded_inverse[:count] = np.linalg.inv(matrices.cpu().numpy()).transpose(
            0, 2, 1
        )
        inverse_cameras = gpu.buffer(padded_inverse.tobytes())
        atlas = gpu.buffer(size=4096 * 4096)
        atlas_uniforms = gpu.buffer(struct.pack("<4I", 4096, 4096, 0, 0))
        bindings = [
            (0, 0, 0, tables),
            (0, 1, 1, atlas),
            (0, 2, 0, atlas_uniforms),
            (0, 3, 1, settings),
        ]
        bindings.extend((1, i, 1, b) for i, b in enumerate(mesh_buffers))
        bindings.extend(
            [
                (2, 0, 2, paint_image),
                (2, 1, 2, depth_image),
                (2, 2, 3, 0),
                (2, 3, 0, cameras),
                (2, 4, 0, inverse_cameras),
            ]
        )
        stage = time.perf_counter()
        project = gpu.pipeline("project", bindings)
        timings["compile_project_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        gpu.call("rd_dispatch", project, 32, 1024)
        timings["project_and_encode_seconds"] = time.perf_counter() - stage
        atlas_image = gpu.call("rd_image", 4096, 4096, 1, 145)
        gpu.call("rd_copy_image", atlas, atlas_image, 0)
        # Render the exact saved camera using the compressed atlas, without mesh export.
        preview_size = 1080
        world = torch.stack(
            (
                positions[:, 0],
                -positions[:, 2],
                positions[:, 1],
                torch.ones_like(positions[:, 0]),
            ),
            dim=1,
        )
        raster, _ = dr.rasterize(
            context,
            (world @ matrices[0].T).unsqueeze(0).contiguous(),
            indices,
            resolution=(preview_size, preview_size),
            grad_db=False,
        )
        interpolated, _ = dr.interpolate(uv.unsqueeze(0), raster, indices)
        pixels = torch.cat(
            (
                interpolated[0],
                (raster[0, :, :, 3:4] > 0).float(),
                torch.zeros_like(raster[0, :, :, 3:4]),
            ),
            dim=-1,
        ).contiguous()
        preview_uv = gpu.buffer(pixels)
        preview = gpu.buffer(size=preview_size * preview_size * 4)
        render = gpu.pipeline(
            "preview",
            [
                (0, 0, 1, preview_uv),
                (0, 1, 1, preview),
                (0, 2, 2, atlas_image),
                (0, 3, 3, 0),
            ],
            "preview",
        )
        gpu.call(
            "rd_dispatch", render, (preview_size + 7) // 8, (preview_size + 7) // 8
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        gpu.call(
            "rd_write_atlas", atlas, str(output_directory / "atlas-bc7.ktx").encode()
        )
        image = Image.frombytes(
            "RGBA",
            (preview_size, preview_size),
            gpu.read(preview, preview_size * preview_size * 4),
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image.convert("RGB").save(output_directory / "same-camera.webp", quality=90)
        gpu.call(
            "rd_write_mesh",
            *mesh_buffers[:4],
            str(output_directory / "sparse-uv.ply").encode(),
        )
        timings["normal_mode"] = "implicit_flat" if flat_normals else "stored_vertex"
        timings["camera_count"] = count
        timings["backend_seconds"] = time.perf_counter() - started
        (output_directory / "bake-timings.json").write_text(
            json.dumps(timings, indent=2)
        )
        print("BC7_ATLAS_WRITTEN", json.dumps(timings), flush=True)
    finally:
        gpu.close()
