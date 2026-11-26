# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _round_bf16(value):
    return value.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _layer_norm_modulate(
    X,
    UPDATE,
    MOD,
    PREV_MOD,
    RESIDUAL,
    OUT,
    SEQUENCE: tl.constexpr,
    WIDTH: tl.constexpr,
    MOD_STRIDE: tl.constexpr,
    PREV_MOD_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_UPDATE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    offset = row * WIDTH + col
    batch = row // SEQUENCE
    x = tl.load(X + offset, valid, 0).to(tl.float32)
    if HAS_UPDATE:
        gate = tl.load(PREV_MOD + batch * PREV_MOD_STRIDE + 2 * WIDTH + col, valid, 0)
        update = tl.load(UPDATE + offset, valid, 0)
        x = _round_bf16(x + _round_bf16(gate.to(tl.float32) * update.to(tl.float32)))
        tl.store(RESIDUAL + offset, x, valid)

    mean = tl.sum(x, axis=0) / WIDTH
    centered = tl.where(valid, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    normalized = _round_bf16(centered * tl.rsqrt(variance + EPS))

    shift = tl.load(MOD + batch * MOD_STRIDE + col, valid, 0).to(tl.float32)
    scale = tl.load(MOD + batch * MOD_STRIDE + WIDTH + col, valid, 0).to(tl.float32)
    # Preserve eager BF16 rounding for LayerNorm and elementwise modulation.
    result = _round_bf16(normalized * _round_bf16(1.0 + scale)) + shift
    tl.store(OUT + offset, result, valid)


@triton_op("my_qwen_image::norm_modulate", mutates_args={})
def norm_modulate(
    x: torch.Tensor,
    mod: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    batch, seq, width = x.shape
    out = torch.empty_like(x)
    mod_stride = mod.stride(0) if mod.shape[0] > 1 else 0
    wrap_triton(_layer_norm_modulate)[(batch * seq,)](
        x, x, mod, mod, out, out, seq, width, mod_stride, 0, eps,
        triton.next_power_of_2(width), False, enable_fp_fusion=False,
    )
    return out


@triton_op("my_qwen_image::residual_norm_modulate", mutates_args={})
def residual_norm_modulate(
    x: torch.Tensor,
    update: torch.Tensor,
    prev_mod: torch.Tensor,
    mod: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq, width = x.shape
    residual = torch.empty_like(x)
    out = torch.empty_like(x)
    mod_stride = mod.stride(0) if mod.shape[0] > 1 else 0
    prev_mod_stride = prev_mod.stride(0) if prev_mod.shape[0] > 1 else 0
    wrap_triton(_layer_norm_modulate)[(batch * seq,)](
        x, update, mod, prev_mod, residual, out, seq, width, mod_stride,
        prev_mod_stride, eps, triton.next_power_of_2(width), True,
        enable_fp_fusion=False,
    )
    return residual, out


@triton.jit
def _gated_residual(
    X,
    UPDATE,
    MOD,
    OUT,
    SIZE: tl.constexpr,
    SEQUENCE: tl.constexpr,
    WIDTH: tl.constexpr,
    MOD_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < SIZE
    col = offset % WIDTH
    batch = offset // (SEQUENCE * WIDTH)
    x = tl.load(X + offset, valid, 0).to(tl.float32)
    update = tl.load(UPDATE + offset, valid, 0).to(tl.float32)
    gate = tl.load(MOD + batch * MOD_STRIDE + 2 * WIDTH + col, valid, 0).to(tl.float32)
    tl.store(OUT + offset, x + _round_bf16(update * gate), valid)


@triton_op("my_qwen_image::gated_residual", mutates_args={})
def gated_residual(x: torch.Tensor, update: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
    _, seq, width = x.shape
    out = torch.empty_like(x)
    mod_stride = mod.stride(0) if mod.shape[0] > 1 else 0
    wrap_triton(_gated_residual)[(triton.cdiv(x.numel(), 1024),)](
        x, update, mod, out, x.numel(), seq, width, mod_stride, 1024,
        enable_fp_fusion=False,
    )
    return out
