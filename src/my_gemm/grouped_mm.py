# pyright: reportInvalidTypeForm=false

from typing import override

import tilelang
import tilelang.language as T
import torch
from torch.autograd.function import FunctionCtx, once_differentiable


@tilelang.jit(out_idx=[3])
def _grouped_m_kernel():
    M, N, K, E = T.dynamic("M, N, K, E", torch.int32)
    b_strides = T.dynamic("b_expert_stride, b_k_stride, b_n_stride", torch.int64)
    BM, BN, BK = 64, 64, 32

    @T.prim_func
    def kernel(
        a: T.Tensor((M, K), "bfloat16"),
        b: T.StridedTensor((E, K, N), b_strides, "bfloat16"),
        offs: T.Tensor((E,), "int32"),
        c: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), E, threads=128) as (bx, expert):
            a_shared = T.alloc_shared((BM, BK), "bfloat16")
            b_shared = T.alloc_shared((BK, BN), "bfloat16")
            acc = T.alloc_fragment((BM, BN), "float32")
            start = T.if_then_else(expert == 0, 0, offs[expert - 1])
            end = offs[expert]
            col = bx * BN
            for block in T.serial(T.ceildiv(end - start, BM)):
                row = start + block * BM
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                    T.copy(a[row, k * BK], a_shared)
                    T.copy(b[expert, k * BK, col], b_shared)
                    T.gemm(a_shared, b_shared, acc)
                for i, j in T.Parallel(BM, BN):
                    if row + i < end and col + j < N:
                        c[row + i, col + j] = acc[i, j]

            # 最后一个 expert 写入 offs 未覆盖的尾行。
            if expert == E - 1:
                for i in T.serial(end, M):
                    for j in T.Parallel(BN):
                        if col + j < N:
                            c[i, col + j] = 0

    return kernel


@tilelang.jit(out_idx=[3])
def _weight_grad_kernel():
    # 每个 expert 的 dW = dY.T @ X，归约维是分配给该 expert 的 token 行。
    tokens, D_out, D_in, E = T.dynamic("tokens, D_out, D_in, E", torch.int32)
    BM, BN, BK = 64, 64, 32

    @T.prim_func
    def kernel(
        dy: T.Tensor((tokens, D_out), "bfloat16"),
        x: T.Tensor((tokens, D_in), "bfloat16"),
        offs: T.Tensor((E,), "int32"),
        dw: T.Tensor((E, D_out, D_in), "bfloat16"),
    ):
        with T.Kernel(
            T.ceildiv(D_in, BN), T.ceildiv(D_out, BM), E, threads=128
        ) as (bx, by, expert):
            a_shared = T.alloc_shared((BK, BM), "bfloat16")
            b_shared = T.alloc_shared((BK, BN), "bfloat16")
            acc = T.alloc_fragment((BM, BN), "float32")
            start = T.if_then_else(expert == 0, 0, offs[expert - 1])
            end = offs[expert]
            full_tiles = (end - start) // BK
            row, col = by * BM, bx * BN
            T.clear(acc)
            for k in T.Pipelined(full_tiles, num_stages=2):
                T.copy(dy[start + k * BK, row], a_shared)
                T.copy(x[start + k * BK, col], b_shared)
                T.gemm(a_shared, b_shared, acc, transpose_A=True)

            # 归约尾块必须在 expert 边界补零，否则会把下一组的 token 加进当前 dW。
            tail = start + full_tiles * BK
            if tail < end:
                tail_a = T.alloc_shared((BK, BM), "bfloat16")
                tail_b = T.alloc_shared((BK, BN), "bfloat16")
                for i, j in T.Parallel(BK, BM):
                    tail_a[i, j] = T.if_then_else(
                        tail + i < end and row + j < D_out, dy[tail + i, row + j], 0
                    )
                for i, j in T.Parallel(BK, BN):
                    tail_b[i, j] = T.if_then_else(
                        tail + i < end and col + j < D_in, x[tail + i, col + j], 0
                    )
                T.gemm(tail_a, tail_b, acc, transpose_A=True)
            T.copy(acc, dw[expert, row, col])

    return kernel


def _grouped_m(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    if a.size(0) == 0:
        return a.new_empty((0, b.size(2)))
    with torch.cuda.device(a.device):
        return _grouped_m_kernel()(a.contiguous(), b, offs)


def _weight_grad(dy: torch.Tensor, x: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    if x.size(0) == 0:
        return x.new_zeros((offs.numel(), dy.size(1), x.size(1)))
    with torch.cuda.device(x.device):
        return _weight_grad_kernel()(dy.contiguous(), x.contiguous(), offs)


class _GroupedMM(torch.autograd.Function):
    @staticmethod
    @override
    def forward(
        ctx: FunctionCtx, a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(
            a if ctx.needs_input_grad[1] else None,
            b if ctx.needs_input_grad[0] else None,
            offs,
        )
        return _grouped_m(a, b, offs)

    @staticmethod
    @override
    @once_differentiable
    def backward(
        ctx: FunctionCtx, dc: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        a, b, offs = ctx.saved_tensors
        da = _grouped_m(dc, b.mT, offs) if ctx.needs_input_grad[0] else None
        # B = W.mT，所以返回给 B 的梯度是 dW.mT。
        db = _weight_grad(dc, a, offs).mT if ctx.needs_input_grad[1] else None
        return da, db, None


def my_grouped_mm(mat_a: torch.Tensor, mat_b: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
    """用 offs 划分 mat_a 的 token 行，每组与 mat_b 中对应的矩阵相乘，支持一阶梯度。"""
    assert mat_a.is_cuda and mat_a.device == mat_b.device == offs.device
    assert mat_a.dtype == mat_b.dtype == torch.bfloat16
    assert mat_a.ndim == 2 and mat_b.ndim == 3
    assert offs.dtype == torch.int32 and offs.ndim == 1 and offs.is_contiguous()
    tokens, k = mat_a.shape
    experts, weight_k, n = mat_b.shape
    assert experts == offs.numel() and experts > 0
    assert k == weight_k and min(k, n) > 0
    assert (offs[0] >= 0) & (offs[-1] <= tokens) & (offs[1:] >= offs[:-1]).all(), "invalid offs"
    return _GroupedMM.apply(mat_a, mat_b, offs)
