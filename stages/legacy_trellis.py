#!/usr/bin/env python3
"""Legacy Espalier-2 pipeline: pure TRELLIS.2 generation and export.

  python3 stages/legacy_trellis.py <image> <output_dir> [--stage generate|export]
"""
import argparse, json, os, sys, time

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src", "TRELLIS.2"))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("output")
p.add_argument("--pipeline", default="1536_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"])
p.add_argument("--seed", type=int, default=42)
p.add_argument("--tile", type=int, default=None)
p.add_argument("--max-tokens", type=int, default=49152)
p.add_argument("--texture", type=int, default=4096)
p.add_argument("--faces", type=int, default=1000000)
p.add_argument("--no-texture", action="store_true")
p.add_argument("--no-rembg", action="store_true")
p.add_argument("--dinov3", default=None)
p.add_argument("--stage", choices=["generate", "export"], required=True)
a = p.parse_args()


def no_rembg(module):
    """Replaces RMBG-2.0 with a stub, so it's never downloaded; images then need an alpha channel."""
    class NoRembg:
        def __init__(self, *x, **k): pass
        def to(self, *x, **k): return self
        def cuda(self): return self
        def cpu(self): return self
        def __call__(self, *x, **k): raise RuntimeError("--no-rembg needs images with an alpha channel")
    module.BiRefNet = NoRembg


def run_trellis2():
    from PIL import Image
    if a.dinov3:
        import trellis2.modules.image_feature_extractor as fe
        load = fe.DINOv3ViTModel.from_pretrained
        fe.DINOv3ViTModel.from_pretrained = lambda name, *x, **k: load(
            a.dinov3 if name == "facebook/dinov3-vitl16-pretrain-lvd1689m" else name, *x, **k)
    if a.no_rembg:
        import trellis2.pipelines.rembg
        no_rembg(trellis2.pipelines.rembg)
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    t0 = time.time()
    meshes, (_, _, res) = pipeline.run(Image.open(a.image), seed=a.seed, pipeline_type=a.pipeline,
                                       max_num_tokens=a.max_tokens, texture=not a.no_texture, return_latent=True)
    return meshes[0], res, t0, {}


def generate():
    dense = a.pipeline == "1536_cascade" and a.max_tokens > 49152
    tile = a.tile if a.tile is not None else ((96 if dense else 256) if a.pipeline == "1536_cascade" else 0)
    if tile:
        os.environ["TRELLIS_DECODE_TILE"] = str(tile)
        if dense:
            os.environ.setdefault("TRELLIS_DECODE_TILE_SPLIT", "2")
            os.environ.setdefault("TRELLIS_DECODE_TILE_MARGIN", "16")
    import torch
    os.makedirs(a.output, exist_ok=True)
    mesh, res, t0, extra = run_trellis2()
    raw = {"vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(), "resolution": res}
    if not a.no_texture:
        raw.update(coords=mesh.coords.cpu(), attrs=mesh.attrs.cpu())
    torch.save(raw, os.path.join(a.output, "raw.pt"))
    info = {"model": "trellis2", "pipeline": a.pipeline, "texture": not a.no_texture, "seed": a.seed, "tile": tile,
            "max_tokens": a.max_tokens, **extra, "resolution": res, "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "seconds": round(time.time() - t0, 1)}
    json.dump(info, open(os.path.join(a.output, "generate.json"), "w"), indent=1)
    print("generated", info, flush=True)


def export():
    import torch, cumesh, o_voxel
    r = torch.load(os.path.join(a.output, "raw.pt"))
    t0 = time.time()
    m = cumesh.CuMesh()
    m.init(r["vertices"].cuda(), r["faces"].cuda())
    m.simplify(16777216, verbose=False)
    v, f = m.read()
    del m
    layout = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}
    textured = "attrs" in r
    glb = o_voxel.postprocess.to_glb(
        vertices=v, faces=f, attr_volume=r.get("attrs"), coords=r.get("coords"), attr_layout=layout,
        voxel_size=1 / r["resolution"], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=a.faces,
        texture_size=a.texture, remesh=True, remesh_band=1, remesh_project=0, verbose=True, bake=textured)
    glb.export(os.path.join(a.output, "model.glb"), extension_webp=True)
    info = {"seconds": round(time.time() - t0, 1)}
    json.dump(info, open(os.path.join(a.output, "export.json"), "w"), indent=1)
    print("exported", os.path.join(a.output, "model.glb"), info, flush=True)


if a.stage == "generate":
    generate()
else:
    export()
