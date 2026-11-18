# pyright: reportInvalidTypeForm=false

import tilelang
import tilelang.language as T
import torch


BLK_SZ = 128


@tilelang.jit
def _quant_kernel(dtype: str, round_scale: bool):
    B = T.dynamic("B", torch.int32)
    M = T.dynamic("M", torch.int32)
    N = T.dynamic("N", torch.int32)
    MB = T.ceildiv(M, BLK_SZ)
    NB = T.ceildiv(N, BLK_SZ)

    @T.prim_func
    def kernel(
        x: T.Tensor((B, M, N), dtype),
        y: T.Tensor((B, M, N), "float8_e4m3"),
        scales: T.Tensor((B, MB, NB), "float32"),
    ):
        with T.Kernel(NB, MB, B, threads=256) as (col_i, row_i, b_i):

            values = T.alloc_fragment((BLK_SZ * BLK_SZ,), "float32")
            block_max = T.alloc_fragment((1,), "float32")
            for k in T.Parallel(BLK_SZ * BLK_SZ):
                row = row_i * BLK_SZ + k // BLK_SZ
                col = col_i * BLK_SZ + k % BLK_SZ
                values[k] = T.if_then_else(row < M and col < N, x[b_i, row, col], 0)
            T.reduce_absmax(values, block_max, dim=0)

            scale = T.max(block_max[0], 1e-4) / 448.0
            if round_scale:
                # scale 是正的 normal FP32。尾数非零时进位，再清零尾数。
                bits = T.reinterpret(scale, T.uint32)
                rounded = (bits + T.uint32(0x7FFFFF)) & T.uint32(0xFF800000)
                scale = T.reinterpret(rounded, T.float32)

            if T.get_thread_binding() == 0:
                scales[b_i, row_i, col_i] = scale

            for i, j in T.Parallel(BLK_SZ, BLK_SZ):
                row = row_i * BLK_SZ + i
                col = col_i * BLK_SZ + j
                if row < M and col < N:
                    y[b_i, row, col] = T.clamp(values[i * BLK_SZ + j] / scale, -448.0, 448.0)

    return kernel


def my_fp8_quant(
    x: torch.Tensor, *, round_scale: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """对最后两维按 128×128 分块，每块返回一个 FP32 scale，边界块只统计有效元素。"""
    assert x.is_cuda and x.dtype in (torch.bfloat16, torch.float32)
    assert x.ndim >= 2
    M, N = x.shape[-2:]
    MB = (M + BLK_SZ - 1) // BLK_SZ
    NB = (N + BLK_SZ - 1) // BLK_SZ
    y = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = x.new_empty((*x.shape[:-2], MB, NB), dtype=torch.float32)
    if x.numel():
        dtype = "bfloat16" if x.dtype == torch.bfloat16 else "float32"
        with torch.cuda.device(x.device):
            kernel = _quant_kernel(dtype, round_scale)
            kernel(
                x.contiguous().view(-1, M, N),
                y.view(-1, M, N),
                scales.view(-1, MB, NB),
            )
    return y, scales
