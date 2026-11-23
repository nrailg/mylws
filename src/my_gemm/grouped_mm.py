# pyright: reportInvalidTypeForm=false

from typing import override

import tilelang
import tilelang.language as T
import torch
from torch.autograd.function import FunctionCtx, once_differentiable


@tilelang.jit
def _grouped_m_kernel(transpose_b: bool):
    M, N, K, E, out_rows = T.dynamic("M, N, K, E, out_rows", torch.int32)
    lda, ldb, batch_stride = T.dynamic("lda, ldb, batch_stride", torch.int64)
    BM, BN, BK = 64, 64, 32
    b_shape = (E, N, K) if transpose_b else (E, K, N)
    b_tile = (BN, BK) if transpose_b else (BK, BN)

    @T.prim_func
    def kernel(
        a: T.StridedTensor((M, K), (lda, 1), "bfloat16"),
        b: T.StridedTensor(b_shape, (batch_stride, ldb, 1), "bfloat16"),
        offs: T.Tensor((E,), "int32"),
        c: T.StridedTensor((out_rows, N), (T.ceildiv(N, 8) * 8, 1), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), E, threads=128) as (bx, expert):
            a_shared = T.alloc_shared((BM, BK), "bfloat16")
            b_shared = T.alloc_shared(b_tile, "bfloat16")
            acc = T.alloc_fragment((BM, BN), "float32")
            start = T.if_then_else(expert == 0, 0, offs[expert - 1])
            end = offs[expert]
            col = bx * BN
            for block in T.serial(T.ceildiv(end - start, BM)):
                row = start + block * BM
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                    T.copy(a[row, k * BK], a_shared)
                    if transpose_b:
                        T.copy(b[expert, col, k * BK], b_shared)
                    else:
                        T.copy(b[expert, k * BK, col], b_shared)
                    T.gemm(a_shared, b_shared, acc, transpose_B=transpose_b)
                for i, j in T.Parallel(BM, BN):
                    if row + i < end and col + j < N:
                        c[row + i, col + j] = acc[i, j]

    return kernel


@tilelang.jit
def _grouped_k_kernel():
    # 两个输入都以 token 为行，在每组内计算 left.T @ right。
    L, R, M, N, E = T.dynamic("L, R, M, N, E", torch.int32)
    lda, ldb = T.dynamic("lda, ldb", torch.int64)
    ldc = T.ceildiv(N, 8) * 8
    BM, BN, BK = 64, 64, 32

    @T.prim_func
    def kernel(
        left: T.StridedTensor((L, M), (lda, 1), "bfloat16"),
        right: T.StridedTensor((R, N), (ldb, 1), "bfloat16"),
        offs: T.Tensor((E,), "int32"),
        c: T.StridedTensor((E, M, N), (M * ldc, ldc, 1), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), E, threads=128) as (bx, by, expert):
            a_shared = T.alloc_shared((BK, BM), "bfloat16")
            b_shared = T.alloc_shared((BK, BN), "bfloat16")
            acc = T.alloc_fragment((BM, BN), "float32")
            start = T.if_then_else(expert == 0, 0, offs[expert - 1])
            end = offs[expert]
            full_tiles = (end - start) // BK
            row, col = by * BM, bx * BN
            T.clear(acc)
            for k in T.Pipelined(full_tiles, num_stages=2):
                T.copy(left[start + k * BK, row], a_shared)
                T.copy(right[start + k * BK, col], b_shared)
                T.gemm(a_shared, b_shared, acc, transpose_A=True)

            # 归约尾块必须在 expert 边界补零，否则会把下一组的 token 加进当前 dW。
            tail = start + full_tiles * BK
            if tail < end:
                tail_a = T.alloc_shared((BK, BM), "bfloat16")
                tail_b = T.alloc_shared((BK, BN), "bfloat16")
                for i, j in T.Parallel(BK, BM):
                    tail_a[i, j] = T.if_then_else(
                        tail + i < end and row + j < M, left[tail + i, row + j], 0
                    )
                for i, j in T.Parallel(BK, BN):
                    tail_b[i, j] = T.if_then_else(
                        tail + i < end and col + j < N, right[tail + i, col + j], 0
                    )
                T.gemm(tail_a, tail_b, acc, transpose_A=True)
            T.copy(acc, c[expert, row, col])

    return kernel


def _empty_aligned(ref: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    # 保留逻辑列数，物理行步长按 BF16 的 16 字节对齐。
    cols = shape[-1]
    return ref.new_empty((*shape[:-1], (cols + 7) // 8 * 8))[..., :cols]


def _aligned_rows(t: torch.Tensor) -> torch.Tensor:
    if t.stride(-1) != 1 or t.stride(-2) < t.size(-1):
        t = _empty_aligned(t, tuple(t.shape)).copy_(t)
    assert t.data_ptr() % 16 == 0, "expected 16-byte alignment"
    assert t.stride(-2) % 8 == 0, "expected 16-byte row strides"
    if t.ndim == 3:
        assert t.stride(0) % 8 == 0, "expected 16-byte expert strides"
    return t


def _grouped_m(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor, out_rows: int) -> torch.Tensor:
    c = _empty_aligned(a, (out_rows, b.size(2))).zero_()
    if a.size(0) and out_rows:
        transpose_b = b.stride(-1) != 1 and b.stride(-2) == 1
        a = _aligned_rows(a)
        b = _aligned_rows(b.mT if transpose_b else b)
        with torch.cuda.device(a.device):
            _grouped_m_kernel(transpose_b)(a, b, offs, c)
    return c


def _grouped_k(left: torch.Tensor, right: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    c = _empty_aligned(left, (offs.numel(), left.size(1), right.size(1)))
    if left.size(0) == 0 or right.size(0) == 0:
        return c.zero_()
    left, right = _aligned_rows(left), _aligned_rows(right)
    with torch.cuda.device(left.device):
        _grouped_k_kernel()(left, right, offs, c)
    return c


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
        ctx.grouped_m = b.ndim == 3
        if ctx.grouped_m:
            return _grouped_m(a, b, offs, a.size(0))
        ctx.left_tokens, ctx.right_tokens = a.size(1), b.size(0)
        return _grouped_k(a.T, b, offs)

    @staticmethod
    @override
    @once_differentiable
    def backward(
        ctx: FunctionCtx, dc: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        a, b, offs = ctx.saved_tensors
        da = db = None
        if ctx.grouped_m:
            if ctx.needs_input_grad[0]:
                da = _grouped_m(dc, b.mT, offs, dc.size(0))
            if ctx.needs_input_grad[1]:
                # dW 按 Linear 权重的 [E, N, K] 布局产生，B = W.mT。
                db = _grouped_k(dc, a, offs).mT
        else:
            if ctx.needs_input_grad[0]:
                da = _grouped_m(b, dc.mT, offs, ctx.left_tokens).T
            if ctx.needs_input_grad[1]:
                db = _grouped_m(a.T, dc, offs, ctx.right_tokens)
        return da, db, None


def my_grouped_mm(mat_a: torch.Tensor, mat_b: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
    """按累计结束位置执行 CUDA BF16 grouped_mm，支持 2D × 3D 和 2D × 2D 的一阶梯度。"""
    assert mat_a.is_cuda and mat_a.device == mat_b.device == offs.device
    assert mat_a.dtype == mat_b.dtype == torch.bfloat16
    assert mat_a.ndim == 2 and mat_b.ndim in (2, 3)
    assert offs.dtype == torch.int32 and offs.ndim == 1 and offs.is_contiguous()
    assert offs.numel() > 0
    if mat_b.ndim == 3:
        experts, k, n = mat_b.shape
        assert experts == offs.numel() and mat_a.size(1) == k and min(k, n) > 0
        limit = mat_a.size(0)
    else:
        assert min(mat_a.size(0), mat_b.size(1)) > 0
        limit = min(mat_a.size(1), mat_b.size(0))
    assert (offs[0] >= 0) & (offs[-1] <= limit) & (offs[1:] >= offs[:-1]).all(), "invalid offs"
    return _GroupedMM.apply(mat_a, mat_b, offs)
