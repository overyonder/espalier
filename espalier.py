"""over|yonder Espalier-2: Microsoft's TRELLIS.2 and Tencent's Pixal3D image-to-3D pipelines at up to 1536 resolution on a
16 GB GPU.

  espalier <image or views directory> <output directory> [--model trellis2] [--pipeline 1536_cascade] [--seed 42]
           [--tile 256] [--max-tokens 49152] [--texture 4096] [--faces 1000000] [--no-texture] [--fov <degrees>]
           [--no-rembg] [--dinov3 <repository>]

Generates in one process and writes <output>/raw.pt, then exports <output>/model.glb in a fresh process, so the
export has the GPU to itself. Everything else follows the model's own example: example.py for TRELLIS.2, and
inference.py or inference_mv.py for Pixal3D."""
import argparse, json, math, os, subprocess, sys, time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "src", "TRELLIS.2"))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Pixal3D's scripts rotate the exported GLB into its frame
PIXAL3D_ROTATION = [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]


def args():
    p = argparse.ArgumentParser(prog="espalier", description=__doc__.split("\n\n")[0])
    p.add_argument("image", help="input image; for Pixal3D's multi-view mode, a directory of posed views with a "
                                 "transforms.json, as its inference_mv.py takes")
    p.add_argument("output")
    p.add_argument("--model", default="trellis2", choices=["trellis2", "pixal3d"],
                   help="trellis2 (microsoft/TRELLIS.2-4B) or pixal3d (TencentARC/Pixal3D, installed with "
                        "install.sh --pixal3d)")
    p.add_argument("--pipeline", default="1536_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"],
                   help="TRELLIS.2's pipelines; Pixal3D has 1024_cascade and 1536_cascade")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile", type=int, default=None,
                   help="decoder tile edge in voxels (default 256 at the 768 stage for 1536_cascade, or 96 at the 384 "
                        "stage when --max-tokens is above 49152; untiled otherwise; 0 turns tiling off)")
    p.add_argument("--max-tokens", type=int, default=49152,
                   help="the cascade's high-resolution token limit; above it the resolution steps down by 128 "
                        "until the object fits (the models' default 49152). Raising it keeps dense objects at 1536 "
                        "and tiles the decoders from their 384 stage")
    p.add_argument("--texture", type=int, default=4096, help="texture size of the exported GLB")
    p.add_argument("--faces", type=int, default=1000000, help="face target of the exported GLB")
    p.add_argument("--no-texture", action="store_true",
                   help="skip the texture sampling, decoder and bake, and export the untextured geometry")
    p.add_argument("--fov", type=float, default=None,
                   help="Pixal3D single-view: the camera's horizontal field of view in degrees, instead of "
                        "estimating it with MoGe-2")
    p.add_argument("--no-rembg", action="store_true",
                   help="don't load RMBG-2.0; the image then needs an alpha channel")
    p.add_argument("--dinov3", default=None,
                   help="TRELLIS.2: Hugging Face repository to load DINOv3 ViT-L/16 from instead of "
                        "facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--stage", choices=["generate", "export"], help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.model == "pixal3d" and a.pipeline not in ("1024_cascade", "1536_cascade"):
        p.error("Pixal3D has only the 1024_cascade and 1536_cascade pipelines")
    if os.path.isdir(a.image) and a.model != "pixal3d":
        p.error("a views directory needs --model pixal3d")
    return a


def no_rembg(module):
    """Replaces RMBG-2.0 with a stub, so it's never downloaded; images then need an alpha channel."""
    class NoRembg:
        def __init__(self, *x, **k): pass
        def to(self, *x, **k): return self
        def cuda(self): return self
        def cpu(self): return self
        def __call__(self, *x, **k): raise RuntimeError("--no-rembg needs images with an alpha channel")
    module.BiRefNet = NoRembg


def run_trellis2(a):
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


def run_pixal3d(a):
    """Pixal3D's inference.py (one image) or inference_mv.py (a views directory), with its own setup, camera handling
    and sampler settings, in low-VRAM mode so each network is on the GPU only for its stage."""
    import inspect, torch
    from PIL import Image
    sys.path.insert(0, os.path.join(here, "src", "Pixal3D"))
    # Pixal3D's projection conditioner runs NAF's cross-attention in tiles, which its 1024 texture stage needs here
    os.environ.setdefault("NAF_TILE", "256")
    if a.no_rembg:
        import pixal3d.pipelines.rembg
        no_rembg(pixal3d.pipelines.rembg)
    multiview = os.path.isdir(a.image)
    px = __import__("inference_mv" if multiview else "inference")
    defaults = inspect.signature(px.run_inference).parameters
    def sampler(stage):
        return {"steps": defaults[f"{stage}_sampling_steps"].default,
                "guidance_strength": defaults[f"{stage}_guidance_strength"].default,
                "guidance_rescale": defaults[f"{stage}_guidance_rescale"].default,
                "rescale_t": defaults[f"{stage}_rescale_t"].default}
    kwargs = dict(seed=a.seed, sparse_structure_sampler_params=sampler("ss"),
                  shape_slat_sampler_params=sampler("shape_slat"), tex_slat_sampler_params=sampler("tex_slat"),
                  return_latent=True, pipeline_type=a.pipeline, max_num_tokens=a.max_tokens,
                  texture=not a.no_texture)
    info = {}
    if multiview:
        pipeline = px.init_pipeline(px.MODEL_PATH, px.CONFIG_FILE, low_vram=True)
        views = px.load_views(a.image, None, rembg=px.make_rembg(pipeline))
        px.check_main_view(views)
        torch.manual_seed(a.seed)
        t0 = time.time()
        meshes, (_, _, res) = pipeline.run_mv(views, **kwargs)
        info["views"] = list(views["view_names"])
    else:
        pipeline = px.init_pipeline(px.MODEL_PATH, low_vram=True)
        image = pipeline.preprocess_image(Image.open(a.image))
        resolution = defaults["image_resolution"].default
        mesh_scale, extend_pixel = defaults["mesh_scale"].default, defaults["extend_pixel"].default
        if a.fov:
            camera_angle_x = math.radians(a.fov)
            distance = px.distance_from_fov(camera_angle_x, torch.tensor([-1.0, 0.0, 0.0]),
                                            torch.tensor([0 - extend_pixel, resolution - 1 + extend_pixel]),
                                            mesh_scale, resolution)["distance_from_x"]
            camera = {"camera_angle_x": camera_angle_x, "distance": distance, "mesh_scale": mesh_scale}
        else:
            # MoGe-2 reads the preprocessed image from a file, as inference.py does
            path = os.path.join(a.output, "preprocessed.png")
            image.save(path)
            moge = px.load_moge_model(device="cuda")
            camera = px.get_camera_params_wild_moge(path, moge, device="cuda", mesh_scale=mesh_scale,
                                                    extend_pixel=extend_pixel, image_resolution=resolution)
            del moge
            torch.cuda.empty_cache()
        info["fov_degrees"] = round(math.degrees(camera["camera_angle_x"]), 2)
        torch.manual_seed(a.seed)
        t0 = time.time()
        meshes, (_, _, res) = pipeline.run(image, camera_params=camera, preprocess_image=False, **kwargs)
    return meshes[0], res, t0, info


def generate(a):
    # Above the token limit the decoders' 384 stage leaves almost no room when it runs whole, so the tiles start there,
    # with a margin covering the later convolutions' reach in that stage's voxels.
    dense = a.pipeline == "1536_cascade" and a.max_tokens > 49152
    tile = a.tile if a.tile is not None else ((96 if dense else 256) if a.pipeline == "1536_cascade" else 0)
    if tile:
        os.environ["TRELLIS_DECODE_TILE"] = str(tile)
        if dense:
            os.environ.setdefault("TRELLIS_DECODE_TILE_SPLIT", "2")
            os.environ.setdefault("TRELLIS_DECODE_TILE_MARGIN", "16")
    import torch
    os.makedirs(a.output, exist_ok=True)
    mesh, res, t0, extra = (run_pixal3d if a.model == "pixal3d" else run_trellis2)(a)
    raw = {"vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(), "resolution": res,
           "rotation": PIXAL3D_ROTATION if a.model == "pixal3d" else None}
    if not a.no_texture:
        raw.update(coords=mesh.coords.cpu(), attrs=mesh.attrs.cpu())
    torch.save(raw, os.path.join(a.output, "raw.pt"))
    info = {"model": a.model, "pipeline": a.pipeline, "texture": not a.no_texture, "seed": a.seed, "tile": tile,
            "tile_stage": 384 if dense and tile else (768 if tile else None), "max_tokens": a.max_tokens, **extra,
            "resolution": res, "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "seconds": round(time.time() - t0, 1), "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    json.dump(info, open(os.path.join(a.output, "generate.json"), "w"), indent=1)
    print("generated", info, flush=True)


def export(a):
    import numpy as np, torch, cumesh, o_voxel
    r = torch.load(os.path.join(a.output, "raw.pt"))
    t0 = time.time()
    m = cumesh.CuMesh()
    m.init(r["vertices"].cuda(), r["faces"].cuda())
    m.simplify(16777216, verbose=False)   # example.py's mesh.simplify(16777216), the nvdiffrast limit
    v, f = m.read()
    del m
    layout = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}
    # A raw mesh generated without texture has no attribute volume, so it's remeshed and simplified but not baked
    textured = "attrs" in r
    glb = o_voxel.postprocess.to_glb(
        vertices=v, faces=f, attr_volume=r.get("attrs"), coords=r.get("coords"), attr_layout=layout,
        voxel_size=1 / r["resolution"], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=a.faces,
        texture_size=a.texture, remesh=True, remesh_band=1, remesh_project=0, verbose=True, bake=textured)
    if r.get("rotation") is not None:
        glb.apply_transform(np.array(r["rotation"], dtype=np.float64))
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
