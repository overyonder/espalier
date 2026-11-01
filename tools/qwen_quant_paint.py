#!/usr/bin/env python3
"""Qwen-Image-2.1 painting with published FP8 or calibrated NVFP4 weights.

~/code/models/qwen-image-2.1/.venv/bin/python tools/qwen_quant_paint.py fp8 PROMPT OUT IMAGE...
~/code/models/qwen-image-2.1/.venv/bin/python tools/qwen_quant_paint.py batch nvfp4 PROMPT OUTDIR REFERENCE RENDER...
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import contextlib
import functools
import json
import site
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(os.environ.get("ESPALIER_QWEN_ROOT", Path.home() / "code/models/qwen-image-2.1"))
RUNTIME = MODEL_ROOT / ".runtime"
site.addsitedir(str(RUNTIME))
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "nvidia_cutlass_dsl/dsl_packages")]
# These optional precompiled caches belong to vLLM's older FlashInfer. The
# Qwen source build compiles its own kernels and must not discover those caches.
sys.modules["flashinfer_cubin"] = None
sys.modules["flashinfer_jit_cache"] = None

# Runtime and optional-cache exclusions must precede vendor imports.
import torch  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from diffusers.hooks.layerwise_casting import apply_layerwise_casting_hook  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor  # noqa: E402

BASE = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/d26bb61231c349cf6b7896fa83353113880e1ba3"
)
WEIGHTS = MODEL_ROOT / "weights"


def load_published_state(path, *, subclasses=False):
    # This is the TorchAO unflatten operation used by Unsloth's
    # prequant_safetensors.load_prequant_safetensors, rather than a new FP8 GEMM.
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        # safe_open exposes keys(), not the Mapping iteration protocol.
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
    identity = json.loads(metadata["unsloth_metadata"])
    if identity["base_model_id"] != "Qwen/Qwen-Image-2.1":
        raise ValueError(f"Unexpected base model in {path}: {identity}")
    if subclasses:
        from torchao.prototype.safetensors.safetensors_support import (
            unflatten_tensor_state_dict,
        )

        tensors, remaining = unflatten_tensor_state_dict(tensors, metadata)
        if remaining:
            raise ValueError(f"Unconsumed checkpoint tensors: {sorted(remaining)}")
    return tensors


def load_text_encoder():
    state = load_published_state(
        WEIGHTS / "unsloth-fp8/Qwen-Image-2.1-text_encoder-FP8.safetensors"
    )
    config = AutoConfig.from_pretrained(BASE / "text_encoder", local_files_only=True)
    # Keep calculated, non-persistent buffers on CPU; only parameters start on meta.
    with init_empty_weights():
        encoder = Qwen3VLForConditionalGeneration(config)
    encoder.load_state_dict(state, strict=True, assign=True)
    for module in encoder.modules():
        if (
            isinstance(module, torch.nn.Linear)
            and module.weight.dtype == torch.float8_e4m3fn
        ):
            apply_layerwise_casting_hook(
                module, torch.float8_e4m3fn, torch.bfloat16, False
            )
    return encoder.eval()


def load_transformer(quant):
    if quant == "nvfp4":
        from flashinfer.gemm import mm_nvfp4_svdquant, nvfp4_quantize_smooth  # noqa: F401

        directory = WEIGHTS / "procreations-nvfp4"
        sys.path.insert(0, str(directory))
        from nvfp4_runtime import load_nvfp4_transformer

        return load_nvfp4_transformer(directory / "transformer", device="cpu")
    state = load_published_state(
        WEIGHTS / "unsloth-fp8/Qwen-Image-2.1-FP8.safetensors", subclasses=True
    )
    config = QwenImage21Transformer2DModel.load_config(BASE / "transformer")
    with init_empty_weights():
        transformer = QwenImage21Transformer2DModel.from_config(config)
    transformer.load_state_dict(state, strict=True, assign=True)
    # TorchAO's own parametrization exposes packed tensors to nn.Module.to,
    # avoiding the subclass's cross-device aliasing path on Torch 2.11.
    from torchao.utils import unwrap_tensor_subclass

    unwrap_tensor_subclass(transformer)
    return transformer.eval()


def measure_cuda_stage(method, label, module):
    @functools.wraps(method)
    def measured(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.monotonic()
        module.to("cuda")
        try:
            result = method(*args, **kwargs)
        finally:
            module.to("cpu")
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(
            f"{label}: {time.monotonic() - started:.2f}s; CUDA peak allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB",
            flush=True,
        )
        return result

    return measured


def build_pipeline(quant):
    print(f"Loading published {quant} transformer and FP8 text encoder", flush=True)
    transformer = load_transformer(quant)
    encoder = load_text_encoder()
    vae = AutoencoderKLQwenImage21.from_pretrained(
        BASE / "vae", torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe = QwenImage21Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            BASE / "scheduler", local_files_only=True
        ),
        processor=Qwen3VLProcessor.from_pretrained(
            BASE / "processor", local_files_only=True
        ),
        text_encoder=encoder,
        transformer=transformer,
        vae=vae,
    )
    pipe.set_progress_bar_config(disable=True)
    # Keep activations and all computation on CUDA; idle blocks stay in RAM.
    # The stock pipeline still owns conditioning, cache layout and denoising.
    transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=True,
        record_stream=True,
        low_cpu_mem_usage=True,
    )
    pipe.vae.enable_tiling()
    pipe.encode_prompt = measure_cuda_stage(
        pipe.encode_prompt, "text encoding", encoder
    )
    pipe.prepare_latents = measure_cuda_stage(
        pipe.prepare_latents, "condition VAE encoding", vae
    )
    pipe.vae.decode = measure_cuda_stage(pipe.vae.decode, "VAE decoding", vae)
    return pipe


def report_denoising_step(pipe, step, timestep, tensors):
    print(f"Denoising step {step + 1}/{pipe.num_timesteps}", flush=True)
    return tensors


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="1248")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quality", help="Accepted for image-client CLI compatibility")
    parser.add_argument("--model", help="Accepted for image-client CLI compatibility")
    parser.add_argument("quant", choices=("fp8", "nvfp4"))
    parser.add_argument("prompt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("images", nargs="*")
    args_list = sys.argv[1:]
    batch = args_list[:1] == ["batch"]
    args_list = args_list[1:] if batch else args_list
    if args_list[:1] not in (["fp8"], ["nvfp4"]):
        args_list = ["fp8", *args_list]
    args = parser.parse_args(args_list)
    width, height = map(
        int,
        args.size.lower().split("x")
        if "x" in args.size.lower()
        else [args.size, args.size],
    )
    if width % 32 or height % 32:
        parser.error("--size must be divisible by 32")
    pipe = build_pipeline(args.quant)
    prompt = args.prompt.read_text().strip()
    jobs = [(args.output, args.images)]
    if batch:
        reference, *renders = args.images
        jobs = [
            (args.output / Path(render).name, [render, reference]) for render in renders
        ]
    for output, paths in jobs:
        if output.exists():
            raise FileExistsError(output)
        with contextlib.ExitStack() as stack:
            images = [
                stack.enter_context(Image.open(path)).convert("RGBA") for path in paths
            ]
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            result = pipe(
                prompt=prompt,
                image=images or None,
                width=width,
                height=height,
                output_resolution=min(width, height),
                num_inference_steps=args.steps,
                true_cfg_scale=1.0,
                use_kv_cache=False,
                callback_on_step_end=report_denoising_step,
                callback_on_step_end_tensor_inputs=[],
                generator=torch.Generator("cuda").manual_seed(args.seed),
            ).images[0]
            output.parent.mkdir(parents=True, exist_ok=True)
            result.save(output)
            print(
                f"SAVED {output}: {time.monotonic() - started:.2f}s; {args.quant}, {width}x{height}, {args.steps} steps, seed {args.seed}",
                flush=True,
            )


if __name__ == "__main__":
    main()
