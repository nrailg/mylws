# Copyright (c) 2026 Tencent Inc. All rights reserved.
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _gather_kernel(
    X, VALID, Y, INVERSE, H: tl.constexpr, K: tl.constexpr, X_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    slot = tl.load(VALID + row)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + (slot // K) * X_STRIDE + cols, cols < H, 0)
    tl.store(Y + row * H + cols, values, cols < H)
    if block == 0:
        tl.store(INVERSE + slot, row)


@triton.jit
def _reduce_kernel(
    X,
    INVERSE,
    W,
    Y,
    H: tl.constexpr,
    K: tl.constexpr,
    WEIGHTED: tl.constexpr,
    X_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    CONTIGUOUS: tl.constexpr = False
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK, ), 0, tl.float32)
    for k in tl.static_range(K):
        slot = row * K + k
        if CONTIGUOUS:
            index = slot
        else:
            index = tl.load(INVERSE + slot)
        value = tl.load(X + index * X_STRIDE + cols, (index >= 0) & (cols < H), 0)
        if WEIGHTED:
            weight = tl.load(W + slot)
            value = (value.to(tl.float32) * weight).to(X.dtype.element_ty)
        total += value.to(tl.float32)
    tl.store(Y + row * H + cols, total, cols < H)


@triton.jit
def _weighted_backward_kernel(
    GRAD,
    X,
    W,
    VALID,
    DX,
    DW,
    H: tl.constexpr,
    K: tl.constexpr,
    X_STRIDE: tl.constexpr,
    GRAD_STRIDE: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_W: tl.constexpr,
    BLOCK: tl.constexpr,
    CONTIGUOUS: tl.constexpr = False
):
    row = tl.program_id(0)
    if CONTIGUOUS:
        slot = row
    else:
        slot = tl.load(VALID + row)
    cols = tl.arange(0, BLOCK)
    grad = tl.load(GRAD + (slot // K) * GRAD_STRIDE + cols, cols < H, 0).to(tl.float32)
    if NEED_X:
        weight = tl.load(W + slot)
        tl.store(DX + row * H + cols, grad * weight, cols < H)
    if NEED_W:
        value = tl.load(X + row * X_STRIDE + cols, cols < H, 0).to(tl.float32)
        tl.store(DW + slot, tl.sum(grad * value, axis=0))


class _SlotGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, rows: torch.Tensor, valid: torch.Tensor,
                top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        inverse = torch.full((rows.shape[0], top_k), -1, device=rows.device, dtype=torch.int32)
        output = rows.new_empty((valid.numel(), rows.shape[1]))
        if valid.numel():
            _gather_kernel[(valid.numel(), triton.cdiv(rows.shape[1], 512))](
                rows,
                valid,
                output,
                inverse,
                rows.shape[1],
                top_k,
                rows.stride(0),
                512,
            )
        ctx.save_for_backward(inverse)
        ctx.mark_non_differentiable(inverse)
        ctx.hidden = rows.shape[1]
        return output, inverse

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: torch.Tensor, _inverse_grad: None) -> tuple:
        (inverse, ) = ctx.saved_tensors
        grad = grad.contiguous()
        output = grad.new_empty((inverse.shape[0], ctx.hidden))
        if inverse.shape[0]:
            _reduce_kernel[(inverse.shape[0], triton.cdiv(ctx.hidden, 256))](
                grad,
                inverse,
                inverse,
                output,
                ctx.hidden,
                inverse.shape[1],
                False,
                grad.stride(0),
                256,
            )
        return output, None, None


class _WeightedSlotReduce(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, projected: torch.Tensor, weights: torch.Tensor, valid: torch.Tensor,
        inverse: torch.Tensor
    ) -> torch.Tensor:
        output = projected.new_empty((inverse.shape[0], projected.shape[1]))
        if inverse.shape[0]:
            _reduce_kernel[(inverse.shape[0], triton.cdiv(projected.shape[1], 256))](
                projected,
                inverse,
                weights,
                output,
                projected.shape[1],
                inverse.shape[1],
                True,
                projected.stride(0),
                256,
            )
        ctx.save_for_backward(
            projected if ctx.needs_input_grad[1] else None,
            weights if ctx.needs_input_grad[0] else None,
            valid,
        )
        ctx.projected_shape = projected.shape
        ctx.projected_dtype = projected.dtype
        ctx.projected_stride = projected.stride(0)
        ctx.weights_shape = weights.shape
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: torch.Tensor) -> tuple:
        projected, weights, valid = ctx.saved_tensors
        grad = grad.contiguous()
        need_x, need_w = ctx.needs_input_grad[:2]
        dx = grad.new_empty(ctx.projected_shape, dtype=ctx.projected_dtype) if need_x else None
        dw = grad.new_zeros(ctx.weights_shape, dtype=torch.float32) if need_w else None
        if valid.numel():
            _weighted_backward_kernel[(valid.numel(), )](
                grad,
                projected,
                weights,
                valid,
                dx,
                dw,
                ctx.projected_shape[1],
                ctx.weights_shape[1],
                ctx.projected_stride,
                grad.stride(0),
                need_x,
                need_w,
                triton.next_power_of_2(ctx.projected_shape[1]),
            )
        return dx, dw, None, None


class _ContiguousWeightedReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, projected: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        output = projected.new_empty((weights.shape[0], projected.shape[1]))
        if weights.shape[0]:
            _reduce_kernel[(weights.shape[0], triton.cdiv(projected.shape[1], 256))](
                projected,
                weights,
                weights,
                output,
                projected.shape[1],
                weights.shape[1],
                True,
                projected.stride(0),
                256,
                CONTIGUOUS=True,
            )
        ctx.save_for_backward(
            projected if ctx.needs_input_grad[1] else None,
            weights if ctx.needs_input_grad[0] else None,
        )
        ctx.projected_shape = projected.shape
        ctx.projected_dtype = projected.dtype
        ctx.projected_stride = projected.stride(0)
        ctx.weights_shape = weights.shape
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: torch.Tensor) -> tuple:
        projected, weights = ctx.saved_tensors
        grad = grad.contiguous()
        need_x, need_w = ctx.needs_input_grad[:2]
        dx = grad.new_empty(ctx.projected_shape, dtype=ctx.projected_dtype) if need_x else None
        dw = grad.new_empty(ctx.weights_shape, dtype=torch.float32) if need_w else None
        if ctx.projected_shape[0]:
            _weighted_backward_kernel[(ctx.projected_shape[0], )](
                grad,
                projected,
                weights,
                weights,
                dx,
                dw,
                ctx.projected_shape[1],
                ctx.weights_shape[1],
                ctx.projected_stride,
                grad.stride(0),
                need_x,
                need_w,
                triton.next_power_of_2(ctx.projected_shape[1]),
                CONTIGUOUS=True,
            )
        return dx, dw


def weighted_reduce_contiguous(projected: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Fuse weighting and top-k reduction for token-major expert output rows.

    Parameters
    ----------
    projected : torch.Tensor
        CUDA BF16/FP32 ``[N*K, H]`` with contiguous columns.
    weights : torch.Tensor
        Contiguous same-device FP32 ``[N, K]`` routing weights.

    Returns
    -------
    torch.Tensor
        ``[N, H]`` with per-slot projected-dtype rounding before FP32 summation.
        Supports first derivatives and checkpoint/offload without weighted temporaries.

    Raises
    ------
    ValueError
        If device, dtype, shape or layout does not match the input contract.
    """
    if (
        not projected.is_cuda or projected.dtype not in (torch.bfloat16, torch.float32) or
        projected.ndim != 2 or projected.shape[1] <= 0 or projected.stride(1) != 1 or
        weights.ndim != 2 or weights.dtype != torch.float32 or not weights.is_contiguous() or
        weights.device != projected.device or weights.shape[1] <= 0 or
        projected.shape[0] != weights.numel()
    ):
        raise ValueError("weighted_reduce_contiguous requires matching CUDA rows and FP32 weights")
    return _ContiguousWeightedReduce.apply(projected, weights)


def gather_slots(rows: torch.Tensor, valid: torch.Tensor,
                 top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather expert-sorted slots with a deterministic input-gradient reduction.

    Parameters
    ----------
    rows : torch.Tensor
        CUDA BF16/FP32 ``[N, H]`` with contiguous columns.
    valid : torch.Tensor
        Contiguous int64 unique slot indices into ``[N, K]``, in expert order.
    top_k : int
        Width K of the receive-side routing table.

    Returns
    -------
    tuple of torch.Tensor
        Expanded ``[S, H]`` rows and nondifferentiable int32 ``[N, K]`` inverse map.
        Supports empty inputs, checkpoint/offload and first derivatives.

    Raises
    ------
    ValueError
        If device, dtype, shape or layout does not match the input contract.
    """
    if not rows.is_cuda or rows.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("gather_slots requires BF16/FP32 CUDA rows")
    if rows.ndim != 2 or rows.shape[1] <= 0 or rows.stride(1) != 1 or top_k <= 0:
        raise ValueError("gather_slots requires contiguous columns and positive top_k")
    if (
        valid.ndim != 1 or valid.dtype != torch.int64 or not valid.is_contiguous() or
        valid.device != rows.device
    ):
        raise ValueError("gather_slots requires same-device contiguous int64 slot indices")
    return _SlotGather.apply(rows, valid, top_k)


def weighted_reduce_slots(
    projected: torch.Tensor, weights: torch.Tensor, valid: torch.Tensor, inverse: torch.Tensor
) -> torch.Tensor:
    """Fuse routing weights and slot reduction without an expanded weighted buffer.

    Parameters
    ----------
    projected : torch.Tensor
        CUDA BF16/FP32 ``[S, H]`` with contiguous columns, matching gather order.
    weights : torch.Tensor
        Contiguous FP32 ``[N, K]`` routing weights.
    valid, inverse : torch.Tensor
        The slot indices and inverse map used by :func:`gather_slots`.

    Returns
    -------
    torch.Tensor
        ``[N, H]``. Each weighted slot rounds to projected dtype before FP32 summation.
        Supports first derivatives, including router weights and empty experts.

    Raises
    ------
    ValueError
        If device, dtype, shape or layout does not match the input contract.
    """
    if (
        not projected.is_cuda or projected.dtype not in (torch.bfloat16, torch.float32) or
        projected.ndim != 2 or projected.shape[1] <= 0 or projected.stride(1) != 1
    ):
        raise ValueError(
            "weighted_reduce_slots requires BF16/FP32 CUDA rows with contiguous columns"
        )
    if (
        weights.ndim != 2 or weights.shape[1] <= 0 or weights.shape != inverse.shape or
        weights.dtype != torch.float32 or not weights.is_contiguous() or
        weights.device != projected.device or inverse.device != projected.device or
        inverse.dtype != torch.int32 or not inverse.is_contiguous() or valid.ndim != 1 or
        valid.device != projected.device or valid.dtype != torch.int64 or
        not valid.is_contiguous() or valid.numel() != projected.shape[0]
    ):
        raise ValueError("weighted_reduce_slots routing metadata does not match projected rows")
    return _WeightedSlotReduce.apply(projected, weights, valid, inverse)
