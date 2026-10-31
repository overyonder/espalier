"""Load retained object geometry once per round; image bytes stay compressed on disk.

The saved-painting route preserves approved geometry and UVs.
"""

import argparse
import csv
import json
import mmap
import struct
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cumesh
import numpy as np
import nvdiffrast.torch as dr
import torch
from PIL import Image
from plyfile import PlyData
from resident import camera_matrices, run
from resident_vk import bake

GENERATOR = Path(__file__).resolve().parents[2]


def paint_missing(reference, depth_paths, destinations):
    client = Path.home() / ".claude/skills/image-generation/scripts/chatgpt_image.py"
    prompt = Path(__file__).with_name("paint-prompt.txt")

    def generate(depth, destination):
        if destination.is_file() and destination.stat().st_size > 0:
            print("REUSE_PAINT", destination, flush=True)
            return
        pending = destination.with_suffix(".pending.png")
        print("START_PAINT", destination, flush=True)
        result = subprocess.run(
            [
                sys.executable,
                str(client),
                str(prompt),
                str(pending),
                str(depth),
                str(reference),
                "--size",
                "1024x1024",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        if result.returncode:
            raise RuntimeError(
                f"Painting {destination} failed: {result.stdout.strip()}"
            )
        with Image.open(pending) as image:
            if image.width != image.height:
                raise ValueError(f"Painting is not square: {pending}")
            image.verify()
        pending.replace(destination)
        print("FINISHED_PAINT", destination, result.stdout.strip(), flush=True)

    with ThreadPoolExecutor(max_workers=13) as workers:
        list(workers.map(generate, depth_paths, destinations))


def load_geometry(path, with_uv):
    if path.suffix == ".ply":
        data = PlyData.read(path, known_list_len={"face": {"vertex_indices": 3}})
        vertices = torch.as_tensor(
            np.stack([data["vertex"][a] for a in ("x", "y", "z")], axis=1),
            device="cuda",
        )
        faces = torch.as_tensor(
            np.array(data["face"]["vertex_indices"], copy=True),
            device="cuda",
            dtype=torch.int32,
        )
        if not with_uv:
            return vertices, faces
        normals = torch.as_tensor(
            np.stack([data["vertex"][a] for a in ("nx", "ny", "nz")], axis=1),
            device="cuda",
        )
        uv = torch.as_tensor(
            np.stack([data["vertex"][a] for a in ("s", "t")], axis=1), device="cuda"
        )
        settings = path.with_name("bake-timings.json")
        flat = (
            settings.is_file()
            and json.loads(settings.read_text()).get("normal_mode") == "implicit_flat"
        )
        return vertices, normals, uv, faces, flat
    with (
        path.open("rb") as source,
        mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        magic, version, length = struct.unpack_from("<3I", mapped)
        if magic != 0x46546C67 or version != 2 or length != len(mapped):
            raise ValueError(f"Invalid GLB: {path}")
        json_size, kind = struct.unpack_from("<2I", mapped, 12)
        if kind != 0x4E4F534A:
            raise ValueError("First GLB chunk must be JSON")
        document = json.loads(mapped[20 : 20 + json_size])
        binary_offset = 28 + json_size
        if any(
            any(k in n for k in ("matrix", "translation", "rotation", "scale"))
            for n in document.get("nodes", [])
        ):
            raise ValueError(f"Node transforms require scene evaluation: {path}")

        def accessor(index):
            a = document["accessors"][index]
            v = document["bufferViews"][a["bufferView"]]
            if "sparse" in a:
                raise ValueError(f"Sparse accessor requires expansion: {path}")
            components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
            dtype = np.dtype(
                {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
            )
            offset = binary_offset + v.get("byteOffset", 0) + a.get("byteOffset", 0)
            stride = v.get("byteStride", dtype.itemsize * components)
            return np.ndarray(
                (a["count"], components),
                dtype=dtype,
                buffer=mapped,
                offset=offset,
                strides=(stride, dtype.itemsize),
            ).copy()

        positions = []
        indices = []
        normal_arrays = []
        uv_arrays = []
        vertex_offset = 0
        flat = False
        for mesh in document["meshes"]:
            for primitive in mesh["primitives"]:
                if primitive.get("mode", 4) != 4:
                    raise ValueError("Only triangle primitives are object surfaces")
                attrs = primitive["attributes"]
                p = accessor(attrs["POSITION"])
                positions.append(p)
                f = (
                    accessor(primitive["indices"]).reshape(-1, 3).astype("i4")
                    if "indices" in primitive
                    else np.arange(len(p), dtype="i4").reshape(-1, 3)
                )
                indices.append(f + vertex_offset)
                vertex_offset += len(p)
                if with_uv:
                    uv_arrays.append(accessor(attrs["TEXCOORD_0"]))
                    if "NORMAL" in attrs:
                        normal_arrays.append(accessor(attrs["NORMAL"]))
                    else:
                        flat = True
        vertices = torch.as_tensor(
            np.concatenate(positions).astype("f4", copy=False), device="cuda"
        )
        faces = torch.as_tensor(
            np.concatenate(indices), device="cuda", dtype=torch.int32
        )
        if not with_uv:
            return vertices, faces
        if flat:
            mesh = cumesh.CuMesh()
            mesh.init(vertices, faces)
            mesh.compute_vertex_normals()
            normals = mesh.read_vertex_normals()
        else:
            normals = torch.as_tensor(
                np.concatenate(normal_arrays).astype("f4", copy=False), device="cuda"
            )
        uv = torch.as_tensor(
            np.concatenate(uv_arrays).astype("f4", copy=False), device="cuda"
        )
        return vertices, normals, uv, faces, flat


def fit_cameras(vertices):
    # Eight orbit, four angled top and three angled underside views.
    import trimesh

    pts_cpu = vertices.cpu().numpy()
    if len(pts_cpu) > 4000:
        idx = np.linspace(0, len(pts_cpu) - 1, 4000, dtype=int)
        pts_cpu = pts_cpu[idx]
    points = np.stack((pts_cpu[:, 0], -pts_cpu[:, 2], pts_cpu[:, 1]), axis=1)
    template = json.loads(
        (GENERATOR / "tools/cams_template.json").read_text()
    )
    cameras = []
    for item in template["cams"]:
        if item["name"] == "a000e85":
            continue
        rotation = np.asarray(item["matrix"]).copy()
        rotation[:3, 3] = 0
        fov = np.degrees(2 * np.arctan(item["sensor"] / (2 * item["lens"])))
        transform = trimesh.scene.cameras.look_at(
            points, [fov, fov], rotation=rotation, pad=1.03
        )
        cameras.append(
            {
                "name": item["name"],
                "matrix": transform.tolist(),
                "lens": item["lens"],
                "sensor": item["sensor"],
                "shift": [0, 0],
            }
        )
    cameras.extend(bottom_cameras(vertices, template))
    return {"res": template["res"], "cams": cameras}


def bottom_cameras(vertices, camera_data):
    """Three underside views, 120 degrees apart and 50 degrees below level."""
    import trimesh

    pts_cpu = vertices.cpu().numpy()
    if len(pts_cpu) > 4000:
        idx = np.linspace(0, len(pts_cpu) - 1, 4000, dtype=int)
        pts_cpu = pts_cpu[idx]
    points = np.stack((pts_cpu[:, 0], -pts_cpu[:, 2], pts_cpu[:, 1]), axis=1)
    source = camera_data["cams"][2]
    fov = np.degrees(2 * np.arctan(source["sensor"] / (2 * source["lens"])))
    cameras = []
    for azimuth in (0, 120, 240):
        a, e = np.radians([azimuth, -50])
        back = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
        right = np.cross([0, 0, 1], back)
        right /= np.linalg.norm(right)
        rotation = np.eye(4)
        rotation[:3, :3] = np.stack((right, np.cross(back, right), back), axis=1)
        transform = trimesh.scene.cameras.look_at(
            points, [fov, fov], rotation=rotation, pad=1.03
        )
        cameras.append(
            {
                "name": f"a{azimuth:03}em50",
                "matrix": transform.tolist(),
                "lens": source["lens"],
                "sensor": source["sensor"],
                "shift": [0, 0],
            }
        )
    return cameras


def bottom_camera(vertices, camera_data):
    """Fit the underside with the same camera-fitting API as the other views."""
    import trimesh

    bounds = torch.stack((vertices.amin(0), vertices.amax(0))).cpu().numpy()
    corners = trimesh.bounds.corners(bounds)
    points = np.stack((corners[:, 0], -corners[:, 2], corners[:, 1]), axis=1)
    source = camera_data["cams"][0]
    fov = np.degrees(2 * np.arctan(source["sensor"] / (2 * source["lens"])))
    rotation = np.diag([-1.0, 1.0, -1.0, 1.0])
    transform = trimesh.scene.cameras.look_at(
        points, [fov, fov], rotation=rotation, pad=1 / 0.88
    )
    return {
        "name": "a000em90",
        "matrix": transform.tolist(),
        "lens": source["lens"],
        "sensor": source["sensor"],
        "shift": [0, 0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--saved-only", action="store_true")
    parser.add_argument("--include-drafts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generate-paintings", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    jobs_path = args.jobs if args.jobs.is_absolute() else GENERATOR / args.jobs
    with jobs_path.open() as source:
        jobs = [
            j
            for j in csv.DictReader(source, delimiter="\t")
            if j["source_pipeline"] == "final" or args.include_drafts
        ]

    def saved_cameras(job):
        path = GENERATOR / (job["cameras"] or (job["output"] + "/cameras.json"))
        return json.loads(path.read_text()) if path.is_file() else None

    def saved_paintings(job):
        cameras = saved_cameras(job)
        if cameras is None:
            return []
        return [
            GENERATOR
            / c.get(
                "painting",
                str(
                    Path(job["paintings"] or job["new_paintings"])
                    / (c["name"] + ".png")
                ),
            )
            for c in cameras["cams"]
        ]

    if args.saved_only:
        jobs = [
            j
            for j in jobs
            if saved_paintings(j) and all(p.is_file() for p in saved_paintings(j))
        ]
    context = dr.RasterizeCudaContext()
    started = time.perf_counter()
    for index, job in enumerate(jobs, 1):
        output = GENERATOR / job["output"]
        if (
            not args.overwrite
            and (output / "bake-timings.json").is_file()
            and (output / "atlas-bc7.ktx").stat().st_size == 16777284
            and (output / "sparse-uv.ply").is_file()
        ):
            print("ALREADY_FINISHED", index, len(jobs), job["id"], flush=True)
            continue
        item_started = time.perf_counter()
        print("START_OBJECT", index, len(jobs), job["id"], flush=True)
        geometry = GENERATOR / job["mesh"]
        cameras = saved_cameras(job)
        paintings = saved_paintings(job)
        if paintings and all(p.is_file() for p in paintings):
            if (output / "sparse-uv.ply").is_file():
                geometry = output / "sparse-uv.ply"
            vertices, normals, uv, faces, flat = load_geometry(geometry, True)
            future = Future()
            future.set_result((vertices, normals, uv, faces))
            bake(
                future,
                camera_matrices(cameras, vertices.device),
                cameras,
                paintings,
                output,
                context,
                flat_normals=flat,
            )
            del future, vertices, normals, uv, faces
        else:
            if job["dense"]:
                geometry = GENERATOR / job["dense"]
            vertices, faces = load_geometry(geometry, False)
            if cameras is None:
                cameras = fit_cameras(vertices)
            run(
                vertices,
                faces,
                cameras,
                GENERATOR / job["depths"],
                GENERATOR / (job["paintings"] or job["new_paintings"]),
                output,
                paint=(partial(paint_missing, GENERATOR / job["reference"]))
                if args.generate_paintings
                else None,
            )
            del vertices, faces
        (output / "cameras.json").write_text(json.dumps(cameras, indent=2))
        print(
            "FINISHED_OBJECT",
            index,
            len(jobs),
            job["id"],
            time.perf_counter() - item_started,
            flush=True,
        )
    print("ALL_OBJECTS_FINISHED", len(jobs), time.perf_counter() - started, flush=True)


if __name__ == "__main__":
    main()
