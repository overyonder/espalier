"""Upscale camera paintings with the Real-ESRGAN SDK before atlas projection."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as torchvision_functional

def main():
    # BasicSR 1.4.2 imports this removed torchvision module for rgb_to_grayscale.
    sys.modules.setdefault("torchvision.transforms.functional_tensor", torchvision_functional)
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--factor", type=int, choices=[2, 4], default=2)
    parser.add_argument("--overwrite", action="store_true", help="recompute existing outputs")
    parser.add_argument("--cameras", type=Path, help="upscale only views named in this saved camera rig")
    parser.add_argument("--tile", type=int, default=512)
    shape = Path(__file__).resolve().parents[1] / "src/Hunyuan3D-2.1"
    parser.add_argument("--weights", type=Path, default=os.environ.get(
        "REALESRGAN_WEIGHTS", shape / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for painting upscaling")
    if not args.weights.is_file():
        parser.error(f"Missing RealESRGAN_x4plus weights: {args.weights}")
    if args.source.resolve() == args.dest.resolve():
        parser.error("Preserve source paintings by choosing a separate destination")
    if args.source.is_dir():
        sources = ([args.source / (camera["name"] + ".png")
                    for camera in json.loads(args.cameras.read_text())["cams"]]
                   if args.cameras else sorted(args.source.glob("*.png")))
        args.dest.mkdir(parents=True, exist_ok=True)
    else:
        sources = [args.source]
    if not sources:
        parser.error("No painting PNGs found")
    torch.set_num_threads(4)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=str(args.weights), model=model,
                           tile=args.tile, tile_pad=16, pre_pad=0, half=True)
    for index, source in enumerate(sources, 1):
        destination = args.dest / source.name if args.source.is_dir() else args.dest
        with Image.open(source) as image:
            rgba = image.convert("RGBA")
            opaque = Image.alpha_composite(Image.new("RGBA", image.size, "black"), rgba)
            pixels = np.ascontiguousarray(np.array(opaque.convert("RGB"))[..., ::-1])
            alpha = rgba.getchannel("A") if "A" in image.getbands() else None
            width, height = image.size
        if (destination.is_file() and not args.overwrite
                and destination.stat().st_mtime_ns >= max(source.stat().st_mtime_ns, args.weights.stat().st_mtime_ns)):
            with Image.open(destination) as saved:
                saved.verify()
            with Image.open(destination) as saved:
                if (saved.size == (width * args.factor, height * args.factor)
                        and saved.mode == ("RGBA" if alpha is not None else "RGB")):
                    print(f"UPSCALE_REUSE {index}/{len(sources)} {destination}", flush=True)
                    continue
        print(f"UPSCALE_START {index}/{len(sources)} {source}", flush=True)
        result, _ = upsampler.enhance(pixels, outscale=4)
        if args.factor == 2:
            result = cv2.resize(result, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
        if result.shape[:2] != (height * args.factor, width * args.factor):
            raise RuntimeError(f"Unexpected upscale dimensions: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix(".pending.png")
        output = Image.fromarray(np.ascontiguousarray(result[..., ::-1]))
        if alpha is not None:
            output.putalpha(alpha.resize(output.size, Image.Resampling.LANCZOS))
        output.save(pending)
        pending.replace(destination)
        print(f"UPSCALE_DONE {index}/{len(sources)} {destination} {width * args.factor}x{height * args.factor}", flush=True)


if __name__ == "__main__":
    main()
