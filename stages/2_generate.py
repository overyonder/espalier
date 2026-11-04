#!/usr/bin/env python3
"""Stage 2: Shape Generation. Generates 3D occupancy field using Hunyuan3D-2.1 flow matching.

  python3 stages/2_generate.py <cutout.png> <out_field.pt> [--steps 100] [--octree 512] [--seed 12345]
"""
import argparse, os, sys, time
# FlashVDM's varying query batches fragment the default allocator, to 11 GiB reserved for 4.6 GiB in use.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from PIL import Image
from jobs import run_jobs

p = argparse.ArgumentParser()
p.add_argument("cutout", nargs="?", help="input cutout PNG")
p.add_argument("out", nargs="?", help="output field tensor (.pt)")
p.add_argument("--batch", help="tab-separated table of the positional paths, holding models across its rows")
p.add_argument("--steps", type=int, default=100, help="inference steps (default: 100)")
p.add_argument("--octree", type=int, default=512, help="octree resolution (default: 512)")
p.add_argument("--seed", type=int, default=12345, help="sampling seed (default: 12345)")
a = p.parse_args()

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shape_root = os.path.join(here, "src", "Hunyuan3D-2.1")
sys.path.insert(0, os.path.join(shape_root, "hy3dshape"))
sys.path.insert(0, shape_root)

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

print(f"== 2 generate shape (Hunyuan3D-2.1, {a.steps} steps, octree {a.octree}, seed {a.seed})", flush=True)
# The SDK loader holds its checkpoint beside a full-precision parameter copy.
# Use PyTorch's mapped checkpoint and Accelerate's empty parameter construction
# for the large DiT; retain ordinary initialisation for VAE's optional keys.
import yaml
from accelerate import init_empty_weights
from hy3dshape.pipelines import smart_load_model, instantiate_from_config
config_path, checkpoint_path = smart_load_model("tencent/Hunyuan3D-2.1", subfolder="hunyuan3d-dit-v2-1", use_safetensors=False, variant="fp16")
with open(config_path) as configuration:
    config = yaml.safe_load(configuration)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
with init_empty_weights(include_buffers=False):
    model = instantiate_from_config(config["model"])
model.load_state_dict(checkpoint["model"], assign=True)
if any(value.is_meta for value in model.state_dict().values()):
    raise RuntimeError("Shape checkpoint left uninitialised model tensors")
vae = instantiate_from_config(config["vae"])
vae.load_state_dict(checkpoint["vae"], strict=False)
conditioner = instantiate_from_config(config["conditioner"])
if "conditioner" in checkpoint:
    conditioner.load_state_dict(checkpoint["conditioner"])
pipe = Hunyuan3DDiTFlowMatchingPipeline(
    model=model, vae=vae, conditioner=conditioner,
    image_processor=instantiate_from_config(config["image_processor"]),
    scheduler=instantiate_from_config(config["scheduler"]), device="cpu", dtype=torch.float16)
del checkpoint
print("SHAPE_CHECKPOINT_LOADED without duplicate DiT parameter allocation", flush=True)
pipe.device = torch.device("cuda")
encode_cond = pipe.encode_cond

def encode_with_offload(*args, **kwargs):
    pipe.vae.to("cpu")
    pipe.model.to("cpu")
    torch.cuda.empty_cache()
    pipe.conditioner.to("cuda")
    condition = encode_cond(*args, **kwargs)
    pipe.conditioner.to("cpu")
    torch.cuda.empty_cache()
    pipe.model.to("cuda")
    return condition

pipe.encode_cond = encode_with_offload

def load_decoder(module, _inputs):
    pipe.model.to("cpu")
    pipe.conditioner.to("cpu")
    torch.cuda.empty_cache()
    module.to("cuda")

pipe.vae.register_forward_pre_hook(load_decoder)
# Decodes only near the surface, which leaves NaN elsewhere. The octree rounds down to 504 at the requested 512.
pipe.vae.enable_flashvdm_decoder(enabled=True, adaptive_kv_selection=True, topk_mode='mean', mc_algo='dmc')

class FieldSaved(Exception):
    pass

def save_field(latents, **kwargs):
    grid_logits = pipe.vae.volume_decoder(latents, pipe.vae.geo_decoder, **kwargs)
    torch.save({"latents": latents.cpu(), "grid_logits": grid_logits.cpu(), "kwargs": kwargs}, field_output)
    raise FieldSaved(tuple(grid_logits.shape))

pipe.vae.latents2mesh = save_field

def generate_field(cutout, out):
    global field_output
    field_output = out
    started = time.time()
    try:
        with Image.open(cutout) as image:
            pipe(image=image.convert("RGBA"), num_inference_steps=a.steps, octree_resolution=a.octree,
                 generator=torch.manual_seed(a.seed), output_type="trimesh")
    except FieldSaved as field:
        print(f"saved field {field.args[0]} in {time.time() - started:.1f}s -> {out}", flush=True)

run_jobs(a.batch, (a.cutout, a.out), generate_field)
