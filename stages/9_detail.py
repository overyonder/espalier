#!/usr/bin/env python3
"""Stage 9: Restore Detail. Displaces a subdivided copy of the textured model onto the dense surface, for Nanite.

Subdivision interpolates the atlas, so every painted texel stays where it was and nothing is unwrapped or painted again.

  python3 stages/9_detail.py <model.glb> <dense.ply> <out.glb> [--levels 2]
"""
import argparse, json, os, struct, time, numpy as np, torch, trimesh
from cumesh import cuBVH
from jobs import run_jobs

p = argparse.ArgumentParser()
p.add_argument("model", nargs="?", help="textured model (.glb), one primitive")
p.add_argument("dense", nargs="?", help="dense surface in the same frame (.ply)")
p.add_argument("out", nargs="?", help="output model (.glb)")
p.add_argument("--batch", help="tab-separated table of the positional paths, holding models across its rows")
p.add_argument("--levels", type=int, default=2, help="most subdivision levels, each four times the faces, stopping before the dense surface's count (default: 2)")
a = p.parse_args()

print(f"== 9 restore detail ({a.levels} levels)", flush=True)

def read_glb(path):
    with open(path, "rb") as source:
        data = source.read()
    json_size = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + json_size]), data[28 + json_size:]

def accessor(document, binary, index):
    entry = document["accessors"][index]
    view = document["bufferViews"][entry["bufferView"]]
    kind = {5125: "<u4", 5123: "<u2", 5121: "u1", 5126: "<f4"}[entry["componentType"]]
    size = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[entry["type"]]
    offset = view.get("byteOffset", 0) + entry.get("byteOffset", 0)
    return np.frombuffer(binary, kind, entry["count"] * size, offset).reshape(-1, size).copy()

def subdivide(welded, faces, uvs, positions):
    """Splits every face in four. Wedges split per corner pair and positions per welded pair, so atlas seams stay seams."""
    corners = torch.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), 1).reshape(-1, 2)
    wedge_pairs, wedge_of_edge = torch.unique(corners.sort(1).values, dim=0, return_inverse=True)
    position_pairs, position_of_wedge_pair = torch.unique(welded[wedge_pairs].sort(1).values, dim=0, return_inverse=True)
    new_uvs = torch.cat((uvs, uvs[wedge_pairs].mean(1)))
    new_positions = torch.cat((positions, positions[position_pairs].mean(1)))
    new_welded = torch.cat((welded, position_of_wedge_pair + len(positions)))
    mid = (wedge_of_edge + len(uvs)).reshape(-1, 3)
    a0, b0, c0, ab, bc, ca = faces[:, 0], faces[:, 1], faces[:, 2], mid[:, 0], mid[:, 1], mid[:, 2]
    new_faces = torch.cat((torch.stack((a0, ab, ca), 1), torch.stack((ab, b0, bc), 1),
                           torch.stack((ca, bc, c0), 1), torch.stack((ab, bc, ca), 1)))
    return new_welded, new_faces, new_uvs, new_positions

def write_glb(path, document, binary, vertices, normals, uvs, faces):
    """Copies the source's material and image bytes untouched and replaces its geometry."""
    primitive = document["meshes"][0]["primitives"][0]
    chunks, views, accessors = [], [], []
    length = 0
    def add(payload, target=None):
        nonlocal length
        view = {"buffer": 0, "byteOffset": length, "byteLength": len(payload)}
        if target: view["target"] = target
        views.append(view)
        chunks.append(payload + b"\0" * (-len(payload) % 4))
        length += len(payload) + (-len(payload) % 4)
        return len(views) - 1
    def add_accessor(array, kind, component, target, bounds=False):
        entry = {"bufferView": add(array.tobytes(), target), "componentType": component, "count": len(array), "type": kind}
        if bounds: entry.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        accessors.append(entry)
        return len(accessors) - 1
    attributes = {"POSITION": add_accessor(vertices, "VEC3", 5126, 34962, True),
                  "NORMAL": add_accessor(normals, "VEC3", 5126, 34962),
                  "TEXCOORD_0": add_accessor(uvs, "VEC2", 5126, 34962)}
    indices = add_accessor(faces.reshape(-1), "SCALAR", 5125, 34963)
    for image in document.get("images", []):
        view = document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        image["bufferView"] = add(binary[start:start + view["byteLength"]])
    document = {**document, "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": length}],
                "meshes": [{"primitives": [{"attributes": attributes, "indices": indices, "material": primitive.get("material", 0), "mode": 4}]}]}
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    with open(path + ".partial", "wb") as target:
        target.write(struct.pack("<3I", 0x46546C67, 2, 28 + len(encoded) + length))
        target.write(struct.pack("<2I", len(encoded), 0x4E4F534A) + encoded)
        target.write(struct.pack("<2I", length, 0x004E4942) + b"".join(chunks))
    os.replace(path + ".partial", path)

def restore_detail(model_path, dense_path, out):
    started = time.time()
    document, binary = read_glb(model_path)
    primitive = document["meshes"][0]["primitives"][0]
    wedge_positions = torch.tensor(accessor(document, binary, primitive["attributes"]["POSITION"]), dtype=torch.float32).cuda()
    uvs = torch.tensor(accessor(document, binary, primitive["attributes"]["TEXCOORD_0"]), dtype=torch.float32).cuda()
    faces = torch.tensor(accessor(document, binary, primitive["indices"]).reshape(-1, 3).astype("i8"), dtype=torch.long).cuda()
    positions, welded = torch.unique(wedge_positions, dim=0, return_inverse=True)
    dense = trimesh.load(dense_path, process=False, force="mesh")
    for _ in range(a.levels):
        if len(faces) * 4 > len(dense.faces):
            break
        welded, faces, uvs, positions = subdivide(welded, faces, uvs, positions)

    surface = cuBVH(dense.vertices, dense.faces)
    surface_vertices = torch.tensor(np.asarray(dense.vertices), dtype=torch.float32).cuda()
    surface_faces = torch.tensor(np.asarray(dense.faces), dtype=torch.long).cuda()
    displaced = torch.empty_like(positions)
    for start in range(0, len(positions), 262144):
        end = min(start + 262144, len(positions))
        _, face_ids, barycentric = surface.unsigned_distance(positions[start:end], return_uvw=True)
        displaced[start:end] = (surface_vertices[surface_faces[face_ids]] * barycentric[..., None]).sum(1)

    position_faces = welded[faces]
    corners = displaced[position_faces]
    face_normals = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = torch.zeros_like(displaced).index_add_(0, position_faces.reshape(-1), face_normals.repeat_interleave(3, 0))
    normals = torch.nn.functional.normalize(normals, dim=1)
    write_glb(out, document, binary, displaced[welded].cpu().numpy(), normals[welded].cpu().numpy(), uvs.cpu().numpy(),
              faces.cpu().numpy().astype("<u4"))
    print(f"{len(faces)} faces, {len(uvs)} vertices, moved {(displaced - positions).norm(dim=1).mean().item():.5f} on average, "
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB, {time.time() - started:.1f}s -> {out}", flush=True)

run_jobs(a.batch, (a.model, a.dense, a.out), restore_detail)
