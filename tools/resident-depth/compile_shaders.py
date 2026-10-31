"""Assemble shared WGSL sources for Naga's native SPIR-V backend."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
shared = root.parent / "direct-bc7"
output = Path(sys.argv[1])
compiler = sys.argv[2]
source = (shared / "bc7.wgsl").read_text()
encode = source.replace(
    "@group(0) @binding(0) var source_texture",
    "@group(1) @binding(0) var source_texture",
)
project = source.replace(
    "@group(0) @binding(0) var source_texture: texture_2d<f32>;", ""
).replace(
    "textureLoad(source_texture, vec2<u32>(pixel_x, pixel_y), 0)",
    "project_pixel(vec2<u32>(pixel_x,pixel_y))",
)
projection = (
    (shared / "project.wgsl")
    .read_text()
    .replace(",13>", ",16>")
    .replace("i<13", "i<i32(textureNumLayers(paintings))")
    .replace("texture_depth_2d_array", "texture_2d_array<f32>")
    .replace("textureLoad(depths,pixel,i,0);", "textureLoad(depths,pixel,i,0).r;")
    .replace(
        "uv*f32(cameras.metadata.x)/f32(cameras.metadata.y)", "uv*cameras.origins[i].w"
    )
    .replace(
        "normal:vec3<f32>)->vec3<f32>{",
        "normal:vec3<f32>,plane_normal:vec3<f32>)->vec3<f32>{",
    )
    .replace(
        "let z=ndc.z*.5+.5;",
        "let z=visibility_depth(point,plane_normal,i,ndc,pixel,dimensions);",
    )
    .replace(
        "project_surface(point,normal)",
        "project_surface(point,normal,cross(vertex_position(best_vertices.y)-vertex_position(best_vertices.x),vertex_position(best_vertices.z)-vertex_position(best_vertices.x)))",
    )
)
projection += "\n" + (root / "visibility.wgsl").read_text()
for name, text, entry in [
    ("bin", (shared / "bin_uv.wgsl").read_text(), "bin_uv"),
    ("prefix", (root / "prefix.wgsl").read_text(), "prefix"),
    ("encode", encode, "compress_bc7"),
    ("project", project + "\n" + projection, "compress_bc7"),
    ("preview", (root / "preview.wgsl").read_text(), "preview"),
]:
    path = output / (name + ".wgsl")
    path.write_text(text)
    subprocess.run(
        [compiler, str(path), str(output / (name + ".spv")), entry], check=True
    )
