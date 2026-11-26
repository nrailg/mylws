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


@triton.jit
def _qk_norm_rope(
    Q,
    K,
    Q_WEIGHT,
    K_WEIGHT,
    FREQS,
    SEQUENCE: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    FREQ_STRIDE: tl.constexpr,
    EPS_Q: tl.constexpr,
    EPS_K: tl.constexpr,
    HEAD_ROWS: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * HEAD_ROWS + tl.arange(0, HEAD_ROWS)
    is_key = tl.program_id(1) == 1
    col = tl.arange(0, HEAD_BLOCK)
    token = row // HEADS
    head = row % HEADS
    valid = (token[:, None] < SEQUENCE) & (col[None, :] < WIDTH)
    pointer = tl.where(is_key, K, Q)
    row_stride = tl.where(is_key, K_STRIDE, Q_STRIDE)
    offset = token[:, None] * row_stride + head[:, None] * WIDTH + col[None, :]
    x = tl.load(pointer + offset, valid, 0).to(tl.float32)
    weight = tl.load(tl.where(is_key, K_WEIGHT, Q_WEIGHT) + col, col < WIDTH, 0).to(tl.float32)

    variance = tl.sum(x * x, axis=1) / WIDTH
    eps = tl.where(is_key, EPS_K, EPS_Q)
    norm = _round_bf16(x * tl.rsqrt(variance + eps)[:, None])
    norm = _round_bf16(norm * weight[None, :])

    # view_as_real interleaves the real and imaginary components of complex64 RoPE.
    freq_offset = token[:, None] * FREQ_STRIDE + (col[None, :] // 2) * 2
    cos = tl.load(FREQS + freq_offset, valid, 0)
    sin = tl.load(FREQS + freq_offset + 1, valid, 0)
    partner = tl.gather(norm, tl.broadcast_to((col ^ 1)[None, :], (HEAD_ROWS, HEAD_BLOCK)), 1)
    rotated = norm * cos + tl.where(col[None, :] % 2 == 0, -partner, partner) * sin
    tl.store(pointer + offset, rotated, valid)


@triton_op("my_qwen_image::qk_norm_rope_", mutates_args={"q", "k"})
def qk_norm_rope_(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    q_eps: float,
    k_eps: float,
) -> None:
    seq = q.shape[0]
    width = q_weight.numel()
    heads = q.shape[-1] // width
    wrap_triton(_qk_norm_rope)[(triton.cdiv(seq * heads, 4), 2)](
        q, k, q_weight, k_weight, freqs, seq, heads, width, q.stride(0),
        k.stride(0), freqs.stride(0), q_eps, k_eps, 4, triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
