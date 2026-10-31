"""Add one underside painting to retained bakes without changing mesh or UVs."""

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import nvdiffrast.torch as dr
import torch
from batch import GENERATOR, bottom_camera, load_geometry, paint_missing
from resident import camera_matrices, render_depths
from resident_vk import bake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-name", default="resident-bc7-v003")
    args = parser.parse_args()
    jobs_path = args.jobs if args.jobs.is_absolute() else GENERATOR / args.jobs
    with jobs_path.open() as source:
        jobs = [
            job
            for job in csv.DictReader(source, delimiter="\t")
            if job["source_pipeline"] == "final"
            and (args.id is None or job["id"] == args.id)
        ]
    torch.set_grad_enabled(False)
    context = dr.RasterizeCudaContext()
    started = time.perf_counter()
    requests = {}
    failures = []
    with ThreadPoolExecutor(max_workers=13) as painters:
        for job in jobs:
            source = GENERATOR / job["output"]
            output = source.parent / args.output_name
            if (output / "atlas-bc7.ktx").is_file() and (
                output / "bake-timings.json"
            ).is_file():
                print("ALREADY_FINISHED", job["id"], flush=True)
                continue
            cameras = json.loads((source / "cameras.json").read_text())
            if len(cameras["cams"]) != 13:
                raise ValueError("Source must be the retained thirteen-view bake")
            depth = GENERATOR / "renders" / job["id"] / "underside-v001"
            paint = (
                GENERATOR.parents[1]
                / "3-imported/gpt-image"
                / job["id"]
                / "paint/underside-v001"
            )
            depth.mkdir(parents=True, exist_ok=True)
            paint.mkdir(parents=True, exist_ok=True)
            depth_path = depth / "a000em90.png"
            paint_path = paint / "a000em90.png"
            vertices, faces = load_geometry(GENERATOR / job["dense"], False)
            bottom = bottom_camera(vertices, cameras)
            bottom["painting"] = os.path.relpath(paint_path, GENERATOR)
            bottom_data = {"res": cameras["res"], "cams": [bottom]}
            if not depth_path.is_file():
                render_depths(
                    vertices,
                    faces,
                    camera_matrices(bottom_data, vertices.device),
                    cameras["res"],
                    context,
                    [depth_path],
                )
            del vertices, faces
            cameras["cams"].append(bottom)
            output.mkdir(parents=True, exist_ok=True)
            (output / "cameras.json").write_text(json.dumps(cameras, indent=2))
            print("UNDERSIDE_DEPTH_READY", job["id"], flush=True)
            if args.prepare_only:
                continue
            # Keep each small sparse CUDA mesh resident while its painting arrives.
            geometry = load_geometry(source / "sparse-uv.ply", True)
            request = painters.submit(
                paint_missing, GENERATOR / job["reference"], [depth_path], [paint_path]
            )
            requests[request] = (job, source, output, cameras, paint_path, geometry)

        for request in as_completed(requests):
            job, source, output, cameras, paint_path, geometry = requests.pop(request)
            try:
                request.result()
            except (
                RuntimeError,
                ValueError,
                OSError,
                subprocess.TimeoutExpired,
            ) as error:
                failures.append(f"{job['id']}: {error}")
                print("UNDERSIDE_PAINT_FAILED", failures[-1], flush=True)
                continue
            vertices, normals, uv, faces, flat = geometry
            paints = GENERATOR / (job["paintings"] or job["new_paintings"])
            paths = [paints / (c["name"] + ".png") for c in cameras["cams"][:-1]]
            paths.append(paint_path)
            ready = Future()
            ready.set_result((vertices, normals, uv, faces))
            bake(
                ready,
                camera_matrices(cameras, vertices.device),
                cameras,
                paths,
                output,
                context,
                flat_normals=flat,
            )
            if (source / "sparse-uv.ply").read_bytes() != (
                output / "sparse-uv.ply"
            ).read_bytes():
                raise ValueError(f"Mesh or UVs changed: {job['id']}")
            print("UNDERSIDE_FINISHED", job["id"], flush=True)
            del ready, geometry, vertices, normals, uv, faces
    if failures:
        raise RuntimeError("\n".join(failures))
    print(
        "ALL_UNDERSIDES_FINISHED", len(jobs), time.perf_counter() - started, flush=True
    )


if __name__ == "__main__":
    main()
