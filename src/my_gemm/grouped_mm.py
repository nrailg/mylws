# pyright: reportInvalidTypeForm=false

from typing import override

import tilelang
import tilelang.language as T
import torch
from torch.autograd.function import FunctionCtx, once_differentiable


# TileLang grouped GEMM 示例，版权及 MIT 许可见 LICENSE.tilelang。
# https://github.com/tile-ai/tilelang/blob/3f01780f093cc5f6691d106da287a0eeb8e51ad3/examples/grouped_gemm/example_grouped_gemm_bwd.py

_BLOCK_M, _BLOCK_N, _BLOCK_K = 64, 64, 32


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def _grouped_m_kernel(a, b, sizes, starts, padded_starts, transpose_b: bool):
    M = T.dynamic("M", T.int32)
    E, K, N = T.const("E, K, N")
    BM, BN, BK = _BLOCK_M, _BLOCK_N, _BLOCK_K
    a: T.Tensor((M, K), T.bfloat16)
    b: T.Tensor((E, N, K) if transpose_b else (E, K, N), T.bfloat16)
    sizes: T.Tensor((E,), T.int32)
    starts: T.Tensor((E,), T.int32)
    padded_starts: T.Tensor((E,), T.int32)
    c = T.empty((M, N), T.bfloat16)

    with T.Kernel(T.ceildiv(M, BM) + E, T.ceildiv(N, BN), threads=128) as (bx, by):
        a_shared = T.alloc_shared((BM, BK), T.bfloat16)
        b_shared = T.alloc_shared((BN, BK) if transpose_b else (BK, BN), T.bfloat16)
        acc = T.alloc_fragment((BM, BN), T.float32)

        padded_row = bx * BM
        expert = T.alloc_var(T.int32, init=0)
        for i in range(E):
            expert = T.if_then_else(padded_row >= padded_starts[i], i, expert)

        local_row = padded_row - padded_starts[expert]
        row = starts[expert] + local_row
        col = by * BN
        valid_rows = T.min(BM, sizes[expert] - local_row)

        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            T.copy(a[row, k * BK], a_shared)
            if transpose_b:
                T.copy(b[expert, col, k * BK], b_shared)
            else:
                T.copy(b[expert, k * BK, col], b_shared)
            T.gemm(a_shared, b_shared, acc, transpose_B=transpose_b)

        for i, j in T.Parallel(BM, BN):
            if i < valid_rows:
                c[row + i, col + j] = acc[i, j]

    return c


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def _grouped_k_kernel(a, dc, sizes, starts):
    M = T.dynamic("M", T.int32)
    E, K, N = T.const("E, K, N")
    BM, BN, BK = _BLOCK_M, _BLOCK_N, _BLOCK_K
    a: T.Tensor((M, K), T.bfloat16)
    dc: T.Tensor((M, N), T.bfloat16)
    sizes: T.Tensor((E,), T.int32)
    starts: T.Tensor((E,), T.int32)
    db = T.empty((E, K, N), T.bfloat16)

    with T.Kernel(T.ceildiv(K, BM), T.ceildiv(N, BN), E, threads=128) as (bx, by, expert):
        a_shared = T.alloc_shared((BK, BM), T.bfloat16)
        dc_shared = T.alloc_shared((BK, BN), T.bfloat16)
        acc = T.alloc_fragment((BM, BN), T.float32)

        start = starts[expert]
        size = sizes[expert]
        row, col = bx * BM, by * BN
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(size, BK), num_stages=2):
            offset = k * BK
            # 归约尾块在 expert 边界补零。
            for i, j in T.Parallel(BK, BM):
                a_shared[i, j] = T.if_then_else(
                    offset + i < size, a[start + offset + i, row + j], 0
                )
            for i, j in T.Parallel(BK, BN):
                dc_shared[i, j] = T.if_then_else(
                    offset + i < size, dc[start + offset + i, col + j], 0
                )
            T.gemm(a_shared, dc_shared, acc, transpose_A=True)

        T.copy(acc, db[expert, row, col])

    return db


def _grouped_m(
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: torch.Tensor,
    starts: torch.Tensor,
    padded_starts: torch.Tensor,
    *,
    transpose_b: bool = False,
) -> torch.Tensor:
    if a.size(0) == 0:
        return a.new_empty((0, b.size(1 if transpose_b else 2)))
    with torch.cuda.device(a.device):
        return _grouped_m_kernel(a, b, sizes, starts, padded_starts, transpose_b)


def _grouped_k(
    a: torch.Tensor, dc: torch.Tensor, sizes: torch.Tensor, starts: torch.Tensor
) -> torch.Tensor:
    if a.size(0) == 0:
        return a.new_zeros((sizes.numel(), a.size(1), dc.size(1)))
    with torch.cuda.device(a.device):
        return _grouped_k_kernel(a, dc, sizes, starts)


class _GroupedMM(torch.autograd.Function):
    @staticmethod
    @override
    def forward(
        ctx: FunctionCtx, a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        sizes = torch.diff(offs, prepend=offs.new_zeros(1))
        starts = offs - sizes
        # 每组在行数后保留至少一个 padding 行，再按 M tile 对齐。
        padded_sizes = (sizes // _BLOCK_M + 1) * _BLOCK_M
        padded_starts = padded_sizes.cumsum(0, dtype=torch.int32) - padded_sizes
        ctx.save_for_backward(
            a if ctx.needs_input_grad[1] else None,
            b if ctx.needs_input_grad[0] else None,
            sizes,
            starts,
            padded_starts if ctx.needs_input_grad[0] else None,
        )
        return _grouped_m(a, b, sizes, starts, padded_starts)

    @staticmethod
    @override
    @once_differentiable
    def backward(
        ctx: FunctionCtx, dc: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        # md $$C_e = A_e B_e$$
        # md $$\nabla_{A_e} = \nabla_{C_e} B_e^\top$$
        # md $$\nabla_{B_e} = A_e^\top \nabla_{C_e}$$
        a, b, sizes, starts, padded_starts = ctx.saved_tensors
        dc = dc.contiguous()
        da = None
        if ctx.needs_input_grad[0]:
            da = _grouped_m(dc, b, sizes, starts, padded_starts, transpose_b=True)
        db = _grouped_k(a, dc, sizes, starts) if ctx.needs_input_grad[1] else None
        return da, db, None


def my_grouped_mm(mat_a: torch.Tensor, mat_b: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
    """按 offs 的单调累计结束位置分组相乘，最后一项须等于 mat_a 的行数。"""
    assert mat_a.is_cuda and mat_a.device == mat_b.device == offs.device
    assert mat_a.dtype == mat_b.dtype == torch.bfloat16
    assert mat_a.ndim == 2 and mat_b.ndim == 3
    assert offs.dtype == torch.int32 and offs.ndim == 1 and offs.is_contiguous()
    k = mat_a.size(1)
    experts, weight_k, n = mat_b.shape
    assert experts == offs.numel() and experts > 0
    assert k == weight_k and min(k, n) > 0
    return _GroupedMM.apply(mat_a.contiguous(), mat_b.contiguous(), offs)
