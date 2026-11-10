"""Puts one attention kernel behind the names the third-party models call.

ESPALIER_ATTENTION chooses it: flash (FlashAttention-4, exact, the default), sdpa, ck_int8 (Comfy Kitchen, 8-bit) or sa3
(SageAttention 3, FP4). A lossy kernel only sees batches: the many short windows of Hi3D's VAE stay on FlashAttention.
"""
import functools, importlib, os, sys, types

import torch
from torch.nn import functional as F

try:
    import flash_attn
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
except ImportError:  # CuTeDSL, which FlashAttention-4 needs, has no Python 3.15 build yet
    flash_attn = types.ModuleType("flash_attn")
    sys.modules["flash_attn"] = flash_attn
    flash_attn_func = flash_attn_varlen_func = None

BACKEND = os.environ.get("ESPALIER_ATTENTION", "flash" if flash_attn_func else "sdpa")
sdpa = F.scaled_dot_product_attention


def _load_kernel():
    """A kernel takes and returns [B, H, L, D]."""
    if BACKEND == "flash":
        def flash(q, k, v):
            return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))[0].transpose(1, 2)
        return flash
    if BACKEND == "sdpa":
        return sdpa
    if BACKEND == "ck_int8":
        return importlib.import_module("comfy_kitchen").int8_attention
    if BACKEND == "sa3":
        sageattn3 = importlib.import_module("sageattn3").sageattn3_blackwell
        # Chunking bounds its FP32 delta_s, which is 5.25 GiB for Hi3D's batch at 64k tokens.
        return lambda q, k, v: sageattn3(q, k, v, q_chunk_size=4096 if q.shape[2] > 8192 else None)
    raise ValueError(f"Unknown ESPALIER_ATTENTION: {BACKEND}")


kernel = _load_kernel()


def dense_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    """F.scaled_dot_product_attention's signature, for code that calls it with plain batches."""
    if attn_mask is not None or is_causal or scale is not None or dropout_p or q.ndim != 4:
        return sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
    return kernel(q, k, v)


class _Functional(types.ModuleType):
    """torch.nn.functional with its attention replaced, for one module to hold as F."""

    def __getattr__(self, name):
        return dense_attention if name == "scaled_dot_product_attention" else getattr(F, name)


def use_for(module):
    """Sends a module that calls F.scaled_dot_product_attention to the chosen kernel."""
    module.F = _Functional("functional")


def _sequences(cumulative):
    return cumulative.numel() - 1


def _sdpa_varlen(q, k, v, cu_q, cu_k, scale):
    """Exact attention over [N, H, D] sequences, with those of equal length run as one batch."""
    out = torch.empty_like(q)
    groups = {}
    for index, lengths in enumerate(zip(cu_q.diff().tolist(), cu_k.diff().tolist())):
        groups.setdefault(lengths, []).append(index)
    for (length_q, length_k), indices in groups.items():
        take = lambda t, cu, length: torch.stack([t[cu[i]:cu[i] + length] for i in indices]).transpose(1, 2)
        result = sdpa(take(q, cu_q, length_q), take(k, cu_k, length_k), take(v, cu_k, length_k), scale=scale).transpose(1, 2)
        for position, i in enumerate(indices):
            out[cu_q[i]:cu_q[i] + length_q] = result[position]
    return out


def _varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale=None):
    """[N, H, D] sequences. A lossy kernel takes one or more of equal length as a batch, as classifier-free guidance gives."""
    if BACKEND == "flash" or scale is not None or not 1 <= _sequences(cu_q) <= 2 or cu_q.diff().unique().numel() > 1 or cu_k.diff().unique().numel() > 1:
        if flash_attn_varlen_func is None:
            return _sdpa_varlen(q, k, v, cu_q, cu_k, scale)
        return flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_q, max_seqlen_k=max_k, softmax_scale=scale)[0]
    shape = lambda t, length: t.reshape(_sequences(cu_q), length, *t.shape[1:]).transpose(1, 2).contiguous()
    out = kernel(shape(q, max_q), shape(k, max_k), shape(v, max_k))
    return out.transpose(1, 2).reshape(q.shape)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    q, k, v = qkv.unbind(1)
    return _varlen(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, softmax_scale)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, **kwargs):
    k, v = kv.unbind(1)
    return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale)


def varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, **kwargs):
    return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale)


def _dense(q, k, v):
    """[B, L, H, D]."""
    return kernel(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


# FlashAttention 2's interface, which Hi3D and TRELLIS.2 import flash_attn for. FlashAttention-4 keeps its own under .cute.
flash_attn.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
flash_attn.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
flash_attn.flash_attn_varlen_func = varlen_func
flash_attn.flash_attn_qkvpacked_func = lambda qkv, **kwargs: _dense(*qkv.unbind(2))
flash_attn.flash_attn_kvpacked_func = lambda q, kv, **kwargs: _dense(q, *kv.unbind(2))
flash_attn.flash_attn_func = lambda q, k, v, **kwargs: _dense(q, k, v)
print(f"attention: {BACKEND}", file=sys.stderr, flush=True)
