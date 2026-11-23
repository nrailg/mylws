# pyright: reportInvalidTypeForm=false

from typing import override

import tilelang
import tilelang.language as T
import torch
from torch.autograd.function import FunctionCtx, once_differentiable


@tilelang.jit(out_idx=[2])
def _gemm_nn_kernel():
    M = T.dynamic("M", torch.int32)
    N = T.dynamic("N", torch.int32)
    K = T.dynamic("K", torch.int32)
    BM, BN, BK = 128, 128, 64

    @T.prim_func
    def kernel(
        a: T.Tensor((M, K), "bfloat16"),
        b: T.Tensor((K, N), "bfloat16"),
        c: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            a_shared = T.alloc_shared((BM, BK), "bfloat16")
            b_shared = T.alloc_shared((BK, BN), "bfloat16")
            acc = T.alloc_fragment((BM, BN), "float32")
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                T.copy(a[by * BM, k * BK], a_shared)
                T.copy(b[k * BK, bx * BN], b_shared)
                T.gemm(a_shared, b_shared, acc)
            T.copy(acc, c[by * BM, bx * BN])

    return kernel


def _gemm_nn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.contiguous(), b.contiguous()
    assert a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0, "expected 16-byte alignment"
    assert a.shape[1] % 8 == 0 and b.shape[1] % 8 == 0, "expected 16-byte row strides"
    with torch.cuda.device(a.device):
        return _gemm_nn_kernel()(a, b)


class _Gemm(torch.autograd.Function):
    @staticmethod
    @override
    def forward(ctx: FunctionCtx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(
            a if ctx.needs_input_grad[1] else None,
            b if ctx.needs_input_grad[0] else None,
        )
        return _gemm_nn(a, b)

    @staticmethod
    @override
    @once_differentiable
    def backward(
        ctx: FunctionCtx, dc: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        a, b = ctx.saved_tensors
        da = _gemm_nn(dc, b.T) if ctx.needs_input_grad[0] else None
        db = _gemm_nn(a.T, dc) if ctx.needs_input_grad[1] else None
        return da, db


def my_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """计算二维 CUDA BF16 矩阵的 a @ b，支持一阶梯度。"""
    assert a.is_cuda and a.device == b.device
    assert a.dtype == b.dtype == torch.bfloat16
    assert a.ndim == b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.numel() > 0 and b.numel() > 0
    return _Gemm.apply(a, b)
