"""Real-ESRGAN super-resolution upscaler for images, camera views, and GLB textures.

Pure PyTorch implementation of RRDBNet with tiled inference and feather blending.
Runs in fp16 on CUDA using < 1 GB VRAM, fitting comfortably within 16 GB GPUs.

Usage:
    python tools/upscale.py input.png output.png [--factor 2|4] [--tile 512]
    python tools/upscale.py input_dir/ output_dir/ [--factor 2|4]
    python tools/upscale.py model.glb model_upscaled.glb [--factor 2|4]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEFAULT_WEIGHTS = Path(
    os.environ.get(
        "REALESRGAN_WEIGHTS",
        Path.home() / "code/models/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth",
    )
)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with 5 convolutional layers."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block (Real-ESRGAN)."""

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    ):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def load_model(weights_path=DEFAULT_WEIGHTS, device="cuda"):
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    )
    if not Path(weights_path).is_file():
        raise FileNotFoundError(
            f"Real-ESRGAN weights not found at {weights_path}"
        )
    ckpt = torch.load(weights_path, map_location="cpu")
    key = "params_ema" if "params_ema" in ckpt else "params"
    model.load_state_dict(ckpt[key], strict=True)
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda().half()
    return model


def upscale_image(
    img: Image.Image,
    model: nn.Module,
    factor: int = 4,
    tile: int = 512,
    tile_pad: int = 32,
) -> Image.Image:
    """Upscale PIL Image using tiled inference with seamless feather blending.
    
    Keeps large accumulator buffers in CPU memory so VRAM usage stays < 500 MiB
    even when processing 4096 or 8192 images alongside running editor/model processes.
    """
    w, h = img.size
    is_cuda = next(model.parameters()).is_cuda

    src_array = np.array(img.convert("RGB"))
    out_scale = factor
    out_h, out_w = h * out_scale, w * out_scale

    # Accumulator in CPU memory (fp32)
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weights = np.zeros((out_h, out_w, 1), dtype=np.float32)

    stride = tile - 2 * tile_pad
    pad_scale = tile_pad * 4

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile, h)
            x_end = min(x + tile, w)
            y_start = max(0, y_end - tile)
            x_start = max(0, x_end - tile)

            tile_np = src_array[y_start:y_end, x_start:x_end]
            tensor = (
                torch.from_numpy(tile_np)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                / 255.0
            )
            if is_cuda:
                tensor = tensor.cuda().half()

            with torch.no_grad():
                out_tile = model(tensor).clamp(0, 1)

            # Cosine feathering mask at native 4x
            th, tw = out_tile.shape[2], out_tile.shape[3]
            mask = torch.ones((1, 1, th, tw), dtype=out_tile.dtype, device=out_tile.device)

            if y_start > 0:
                ramp = (
                    0.5
                    - 0.5
                    * torch.cos(
                        torch.linspace(
                            0, np.pi, pad_scale, device=out_tile.device, dtype=out_tile.dtype
                        )
                    )
                ).view(1, 1, pad_scale, 1)
                mask[:, :, :pad_scale, :] *= ramp
            if y_end < h:
                ramp = (
                    0.5
                    + 0.5
                    * torch.cos(
                        torch.linspace(
                            0, np.pi, pad_scale, device=out_tile.device, dtype=out_tile.dtype
                        )
                    )
                ).view(1, 1, pad_scale, 1)
                mask[:, :, -pad_scale:, :] *= ramp
            if x_start > 0:
                ramp = (
                    0.5
                    - 0.5
                    * torch.cos(
                        torch.linspace(
                            0, np.pi, pad_scale, device=out_tile.device, dtype=out_tile.dtype
                        )
                    )
                ).view(1, 1, 1, pad_scale)
                mask[:, :, :, :pad_scale] *= ramp
            if x_end < w:
                ramp = (
                    0.5
                    + 0.5
                    * torch.cos(
                        torch.linspace(
                            0, np.pi, pad_scale, device=out_tile.device, dtype=out_tile.dtype
                        )
                    )
                ).view(1, 1, 1, pad_scale)
                mask[:, :, :, -pad_scale:] *= ramp

            if factor != 4:
                tgt_h = (y_end - y_start) * factor
                tgt_w = (x_end - x_start) * factor
                out_tile = F.interpolate(out_tile.float(), size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
                mask = F.interpolate(mask.float(), size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)

            tile_cpu = (out_tile * mask).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            mask_cpu = mask.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

            oy_start = y_start * out_scale
            oy_end = oy_start + tile_cpu.shape[0]
            ox_start = x_start * out_scale
            ox_end = ox_start + tile_cpu.shape[1]

            output[oy_start:oy_end, ox_start:ox_end] += tile_cpu
            weights[oy_start:oy_end, ox_start:ox_end] += mask_cpu

    final_img = (output / np.clip(weights, 1e-5, None)).clip(0, 1)
    return Image.fromarray((final_img * 255).astype(np.uint8))


def upscale_glb(
    glb_in: Path, glb_out: Path, model: nn.Module, factor: int = 2
):
    import trimesh

    mesh = trimesh.load(glb_in, force="mesh", process=False)
    material = mesh.visual.material
    albedo = material.baseColorTexture.convert("RGB")
    print(f"Upscaling GLB atlas from {albedo.size} (factor={factor})...")
    t0 = time.time()
    upscaled = upscale_image(albedo, model, factor=factor)
    print(f"Upscaled atlas to {upscaled.size} in {time.time() - t0:.1f}s")

    sharpened = trimesh.visual.material.PBRMaterial(
        baseColorTexture=upscaled,
        metallicRoughnessTexture=material.metallicRoughnessTexture,
        metallicFactor=material.metallicFactor,
        roughnessFactor=material.roughnessFactor,
    )
    new_mesh = trimesh.Trimesh(
        mesh.vertices,
        mesh.faces,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=sharpened),
    )
    glb_out.parent.mkdir(parents=True, exist_ok=True)
    new_mesh.export(glb_out)
    print(f"Saved upscaled GLB to {glb_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-ESRGAN super-resolution upscaler"
    )
    parser.add_argument("source", type=Path, help="Input image, folder, or GLB")
    parser.add_argument(
        "dest", type=Path, help="Output destination image, folder, or GLB"
    )
    parser.add_argument(
        "--factor",
        type=int,
        choices=[2, 4],
        default=2,
        help="Upscale factor (default: 2)",
    )
    parser.add_argument(
        "--tile", type=int, default=512, help="Tile size (default: 512)"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to RealESRGAN_x4plus.pth",
    )
    args = parser.parse_args()

    model = load_model(args.weights)

    if args.source.is_file() and args.source.suffix.lower() in [".glb", ".gltf"]:
        upscale_glb(args.source, args.dest, model, factor=args.factor)
    elif args.source.is_dir():
        args.dest.mkdir(parents=True, exist_ok=True)
        images = [
            p
            for p in args.source.glob("*.*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
        ]
        print(f"Upscaling {len(images)} images in {args.source}...")
        for img_path in images:
            img = Image.open(img_path)
            res = upscale_image(
                img, model, factor=args.factor, tile=args.tile
            )
            out_path = args.dest / img_path.name
            res.save(out_path)
            print(f"  {img_path.name}: {img.size} -> {res.size}")
    else:
        img = Image.open(args.source)
        res = upscale_image(img, model, factor=args.factor, tile=args.tile)
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        res.save(args.dest)
        print(f"Saved {args.dest} ({img.size} -> {res.size})")


if __name__ == "__main__":
    main()
