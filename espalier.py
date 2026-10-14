"""over|yonder Espalier-2: Microsoft's TRELLIS.2 image-to-3D pipeline at up to 1536 resolution on a 16 GB GPU.

  espalier <image> <output directory> [--pipeline 1536_cascade] [--seed 42] [--tile 256] [--texture 4096]
           [--faces 1000000] [--no-rembg] [--dinov3 <repository>]

Generates in one process and writes <output>/raw.pt, then exports <output>/model.glb in a fresh process, so the
export has the GPU to itself. Everything else follows Microsoft's example.py."""
import argparse, json, os, subprocess, sys, time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "src", "TRELLIS.2"))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def args():
    p = argparse.ArgumentParser(prog="espalier", description=__doc__.split("\n\n")[0])
    p.add_argument("image")
    p.add_argument("output")
    p.add_argument("--pipeline", default="1536_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile", type=int, default=None,
                   help="decoder tile edge in voxels at the second-to-last stage (default 256 for 1536_cascade, "
                        "untiled otherwise; 0 turns tiling off)")
    p.add_argument("--texture", type=int, default=4096, help="texture size of the exported GLB")
    p.add_argument("--faces", type=int, default=1000000, help="face target of the exported GLB")
    p.add_argument("--no-rembg", action="store_true",
                   help="don't load RMBG-2.0; the image then needs an alpha channel")
    p.add_argument("--dinov3", default=None,
                   help="Hugging Face repository to load DINOv3 ViT-L/16 from instead of "
                        "facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--stage", choices=["generate", "export"], help=argparse.SUPPRESS)
    return p.parse_args()


def generate(a):
    tile = a.tile if a.tile is not None else (256 if a.pipeline == "1536_cascade" else 0)
    if tile:
        os.environ["TRELLIS_DECODE_TILE"] = str(tile)
    import torch
    from PIL import Image
    if a.dinov3:
        import trellis2.modules.image_feature_extractor as fe
        load = fe.DINOv3ViTModel.from_pretrained
        fe.DINOv3ViTModel.from_pretrained = lambda name, *x, **k: load(
            a.dinov3 if name == "facebook/dinov3-vitl16-pretrain-lvd1689m" else name, *x, **k)
    if a.no_rembg:
        import trellis2.pipelines.rembg as rembg
        class NoRembg:
            def __init__(self, *x, **k): pass
            def to(self, *x, **k): return self
            def cuda(self): return self
            def cpu(self): return self
            def __call__(self, *x, **k): raise RuntimeError("--no-rembg needs an image with an alpha channel")
        rembg.BiRefNet = NoRembg
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    os.makedirs(a.output, exist_ok=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    t0 = time.time()
    mesh = pipeline.run(Image.open(a.image), seed=a.seed, pipeline_type=a.pipeline)[0]
    res = round(1 / mesh.voxel_size)
    torch.save({"vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(), "coords": mesh.coords.cpu(),
                "attrs": mesh.attrs.cpu(), "resolution": res}, os.path.join(a.output, "raw.pt"))
    info = {"pipeline": a.pipeline, "seed": a.seed, "tile": tile, "resolution": res, "vertices": len(mesh.vertices),
            "faces": len(mesh.faces), "seconds": round(time.time() - t0, 1),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    json.dump(info, open(os.path.join(a.output, "generate.json"), "w"), indent=1)
    print("generated", info, flush=True)


def export(a):
    import torch, cumesh, o_voxel
    r = torch.load(os.path.join(a.output, "raw.pt"))
    t0 = time.time()
    m = cumesh.CuMesh()
    m.init(r["vertices"].cuda(), r["faces"].cuda())
    m.simplify(16777216, verbose=False)   # example.py's mesh.simplify(16777216), the nvdiffrast limit
    v, f = m.read()
    del m
    layout = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}
    glb = o_voxel.postprocess.to_glb(
        vertices=v, faces=f, attr_volume=r["attrs"], coords=r["coords"], attr_layout=layout,
        voxel_size=1 / r["resolution"], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=a.faces,
        texture_size=a.texture, remesh=True, remesh_band=1, remesh_project=0, verbose=True)
    glb.export(os.path.join(a.output, "model.glb"), extension_webp=True)
    info = {"seconds": round(time.time() - t0, 1), "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    json.dump(info, open(os.path.join(a.output, "export.json"), "w"), indent=1)
    print("exported", os.path.join(a.output, "model.glb"), info, flush=True)


if __name__ == "__main__":
    a = args()
    if a.stage == "generate":
        generate(a)
    elif a.stage == "export":
        export(a)
    else:
        # each stage in its own process, and this one never touches the GPU, so the export has the card to itself
        for stage in ("generate", "export"):
            subprocess.run([sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--stage", stage], check=True)
