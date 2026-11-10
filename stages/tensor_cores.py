"""Runs a model's large linear layers on the tensor cores through Comfy Kitchen. Norms, embedders and small layers keep their precision.

ESPALIER_LINEAR chooses the format: fp8, fp4 or none.
"""
import os

import torch
from comfy_kitchen.tensor import QuantizedTensor
from torch import nn
from torch.nn import functional as F

FORMAT = os.environ.get("ESPALIER_LINEAR", "fp8")
LAYOUT = {"fp8": "TensorCoreFP8Layout", "fp4": "TensorCoreNVFP4Layout", "none": None}[FORMAT]
KEEP = ("t_embedder", "x_embedder", "final_layer", "adaLN_modulation", "input_layer", "out_layer", "adaLN")


def _half(x):
    """The tensor cores' quantisers take 16-bit floats."""
    return x.to(torch.bfloat16) if x.dtype == torch.float32 else x


class QuantizedLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.weight = QuantizedTensor.from_float(_half(linear.weight.detach()).contiguous(), LAYOUT)
        self.bias = None if linear.bias is None else _half(linear.bias.detach())

    def _apply(self, fn, recurse=True):
        self.weight = fn(self.weight)
        self.bias = None if self.bias is None else fn(self.bias)
        return self

    def forward(self, x):
        if hasattr(x, "feats"):  # A sparse tensor: the layer it replaces works on its features.
            return x.replace(self.forward(x.feats))
        out = F.linear(QuantizedTensor.from_float(_half(x.flatten(0, -2)), LAYOUT), self.weight, self.bias)
        if isinstance(out, QuantizedTensor):
            out = out.dequantize()
        out = out.to(x.dtype)
        return out.unflatten(0, x.shape[:-1]) if x.ndim > 2 else out


def convert(module, minimum_features=512):
    """Replaces every nn.Linear wider than minimum_features on both sides, once, except under a child named in KEEP."""
    if LAYOUT is None or any(isinstance(child, QuantizedLinear) for child in module.modules()):
        return 0
    return _replace(module, minimum_features)


def _replace(module, minimum_features):
    converted = 0
    for name, child in module.named_children():
        if name in KEEP:
            continue
        if isinstance(child, nn.Linear) and child.weight.ndim == 2 and min(child.in_features, child.out_features) >= minimum_features:
            setattr(module, name, QuantizedLinear(child))
            converted += 1
        else:
            converted += _replace(child, minimum_features)
    return converted
