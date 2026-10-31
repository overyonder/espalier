"""Dense depth conditioning -> retained CUDA reduction -> GPT paints -> BC7 bake.

No intermediate geometry or texture atlas is written. The saved-dense CLI loads
once; generation code can instead call run() with its CUDA tensors directly.
"""

import argparse
import json
import os
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cumesh
import numpy as np
import nvdiffrast.torch as dr
import torch
from PIL import Image


def camera_matrices(data, device):
    result = []
    for camera in data["cams"]:
        world = np.array(camera["matrix"], dtype="f4")
        f = 2 * camera["lens"] / camera["sensor"]
        near = 0.01
        far = 1000.0
        projection = np.array(
            [
                [f, 0, -2 * camera["shift"][0], 0],
                [0, f, -2 * camera["shift"][1], 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0],
            ],
            dtype="f4",
        )
        result.append(projection @ np.linalg.inv(world))
    return torch.as_tensor(np.asarray(result), device=device)


class DepthNormalizer:
    """Replay PyTorch's exact normalization operations with one CUDA launch."""

    def __init__(self, resolution, device):
        self.input = torch.zeros((2, resolution, resolution, 4), device=device)
        self.graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.normalize()
        current.wait_stream(stream)
        with torch.cuda.graph(self.graph):
            self.gray = self.normalize()

    def normalize(self):
        mask = self.input[..., 3] > 0
        depth = self.input[..., 2] * 0.5 + 0.5
        linear = 0.01 * 1000.0 / (1000.0 - depth * (1000.0 - 0.01))
        near = linear.masked_fill(~mask, float("inf")).amin((1, 2), keepdim=True)
        far = linear.masked_fill(~mask, -float("inf")).amax((1, 2), keepdim=True)
        visible = 0.15 + 0.85 * (far - linear) / (far - near).clamp_min(1e-8)
        return (
            (torch.where(mask, visible, 0).clamp(0, 1) * 255)
            .round()
            .to(torch.uint8)
            .flip(1)
        )

    def __call__(self, raster):
        self.input.copy_(raster)
        self.graph.replay()
        return self.gray


_depth_normalizers = {}


def render_depths(vertices, faces, cameras, resolution, context, output=None):
    world = torch.stack(
        (
            vertices[:, 0],
            -vertices[:, 2],
            vertices[:, 1],
            torch.ones_like(vertices[:, 0]),
        ),
        dim=1,
    )
    result = (
        None
        if output is not None
        else torch.empty(
            (len(cameras), resolution, resolution),
            device=vertices.device,
            dtype=torch.float32,
        )
    )
    normalizer = None
    if output is not None:
        key = (vertices.device, resolution)
        if key not in _depth_normalizers:
            _depth_normalizers[key] = DepthNormalizer(resolution, vertices.device)
        normalizer = _depth_normalizers[key]

    def save_gray(pixels, path):
        Image.fromarray(pixels.numpy(), "L").save(path, compress_level=1)

    with ThreadPoolExecutor(max_workers=4) as writers:
        saved = []
        for first in range(0, len(cameras), 2):
            last = min(first + 2, len(cameras))
            batch = cameras[first:last]
            if last - first == 1:
                batch = batch.expand(2, -1, -1)
            clip = torch.matmul(world.unsqueeze(0), batch.transpose(1, 2)).contiguous()
            raster, _ = dr.rasterize(
                context, clip, faces, resolution=(resolution, resolution), grad_db=False
            )
            if output is not None:
                pixels = normalizer(raster).cpu()
                for index in range(last - first):
                    saved.append(
                        writers.submit(save_gray, pixels[index], output[first + index])
                    )
            else:
                mask = raster[..., 3] > 0
                depth = raster[..., 2] * 0.5 + 0.5
                result[first:last].copy_(torch.where(mask, depth, 1)[: last - first])
        for pending in saved:
            pending.result()
    return result


def run(
    vertices,
    faces,
    camera_data,
    depth_directory,
    paint_directory,
    output_directory,
    paint=None,
):
    from resources import ResourceMonitor

    monitor = ResourceMonitor()
    started = time.perf_counter()
    timings = {}
    device = vertices.device
    dense_counts = (len(vertices), len(faces))
    if not 1 <= len(camera_data["cams"]) <= 16:
        raise ValueError("One to sixteen camera transforms are required")
    assert vertices.is_cuda and faces.is_cuda
    torch.set_grad_enabled(False)
    depth_directory.mkdir(parents=True, exist_ok=True)
    paint_directory.mkdir(parents=True, exist_ok=True)
    matrices = camera_matrices(camera_data, device)
    context = dr.RasterizeCudaContext(device=device)
    paths = [depth_directory / (c["name"] + ".png") for c in camera_data["cams"]]
    torch.cuda.synchronize()
    stage = time.perf_counter()
    dense_depths = render_depths(
        vertices, faces, matrices, camera_data["res"], context, paths
    )
    torch.cuda.synchronize()
    timings["dense_depth_seconds"] = time.perf_counter() - stage
    print(
        "DENSE_DEPTHS_READY",
        len(vertices),
        len(faces),
        timings["dense_depth_seconds"],
        flush=True,
    )
    stage = time.perf_counter()
    mesh = cumesh.CuMesh()
    mesh.init(vertices.clone(), faces.clone())

    def clean():
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)

    mesh.fill_holes(max_hole_perimeter=3e-2)
    mesh.simplify(180000)
    clean()
    mesh.simplify(60000)
    clean()
    mesh.unify_face_orientations()
    sparse_vertices, sparse_faces = mesh.read()
    torch.cuda.synchronize()
    timings["reduce_seconds"] = time.perf_counter() - stage
    del dense_depths
    print(
        "WAITING_FOR_GPT_PAINTS",
        len(camera_data["cams"]),
        os.getpid(),
        len(sparse_vertices),
        len(sparse_faces),
        "cuda_bytes",
        torch.cuda.memory_allocated(),
        flush=True,
    )
    expected = [paint_directory / (c["name"] + ".png") for c in camera_data["cams"]]
    wait_started = time.perf_counter()
    if paint is not None:
        paint(paths, expected)
    while not all(p.is_file() and p.stat().st_size > 0 for p in expected):
        if time.perf_counter() - wait_started > 3600:
            raise TimeoutError("GPT paint files did not arrive within one hour")
        time.sleep(0.25)
    timings["paint_wait_seconds"] = time.perf_counter() - wait_started

    def unwrap_mesh():
        torch.cuda.set_device(device)
        stage = time.perf_counter()
        _, unwrapped_faces, uvs, vertex_map = mesh.uv_unwrap(return_vmaps=True)
        base_vertices, _base_faces = mesh.read()
        mesh.compute_vertex_normals()
        base_normals = mesh.read_vertex_normals()
        vertex_map = vertex_map.to(device=device, dtype=torch.long)
        positions = base_vertices[vertex_map].contiguous()
        normals = base_normals[vertex_map].contiguous()
        uv = uvs.to(device=device, dtype=torch.float32).contiguous()
        indices = unwrapped_faces.to(device=device, dtype=torch.int32).contiguous()
        torch.cuda.synchronize()
        timings["unwrap_seconds"] = time.perf_counter() - stage
        return positions, normals, uv, indices

    from resident_vk import bake

    stage = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as worker:
        unwrapped = worker.submit(unwrap_mesh)
        bake(unwrapped, matrices, camera_data, expected, output_directory, context)
        positions, _normals, _uv, indices = unwrapped.result()
    torch.cuda.synchronize()
    timings["bake_and_output_seconds"] = time.perf_counter() - stage
    timings.update(
        total_seconds=time.perf_counter() - started,
        dense_vertices=dense_counts[0],
        dense_faces=dense_counts[1],
        sparse_vertices=len(positions),
        sparse_faces=len(indices),
        torch_peak_vram_mib=torch.cuda.max_memory_allocated() / 1048576,
        peak_ram_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    timings.update(monitor.finish())
    output_directory.mkdir(exist_ok=True, parents=True)
    (output_directory / "timings.json").write_text(json.dumps(timings, indent=2))
    print("ROUND_FINISHED", json.dumps(timings), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dense", type=Path)
    parser.add_argument("cameras", type=Path)
    parser.add_argument("depths", type=Path)
    parser.add_argument("paints", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from plyfile import PlyData

    data = PlyData.read(
        args.dense, mmap="c", known_list_len={"face": {"vertex_indices": 3}}
    )
    vertex_data = np.stack([data["vertex"][axis] for axis in ("x", "y", "z")], axis=1)
    vertices = torch.as_tensor(vertex_data, dtype=torch.float32, device="cuda")
    faces = torch.as_tensor(
        np.array(data["face"]["vertex_indices"], copy=True),
        dtype=torch.int32,
        device="cuda",
    )
    run(
        vertices,
        faces,
        json.loads(args.cameras.read_text()),
        args.depths,
        args.paints,
        args.output,
    )
