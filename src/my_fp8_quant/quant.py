# pyright: reportInvalidTypeForm=false

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def _quant_kernel(width: int, group_size: int, dtype: str, round_scale: bool):
    rows = T.dynamic("rows")
    tile_rows = 16

    @T.prim_func
    def kernel(
        x: T.Tensor((rows, width), dtype),
        y: T.Tensor((rows, width), "float8_e4m3"),
        scales: T.Tensor((rows, width // group_size), "float32"),
    ):
        with T.Kernel(width // group_size, T.ceildiv(rows, tile_rows), threads=128) as (gx, gy):
            values = T.alloc_fragment((tile_rows, group_size), "float32")
            maxima = T.alloc_fragment((tile_rows,), "float32")
            scale = T.alloc_fragment((tile_rows,), "float32")

            for i, j in T.Parallel(tile_rows, group_size):
                row = gy * tile_rows + i
                values[i, j] = T.if_then_else(row < rows, x[row, gx * group_size + j], 0)
            T.reduce_absmax(values, maxima, dim=1)
            for i in T.Parallel(tile_rows):
                s = T.max(maxima[i], 1e-4) * (1.0 / 448.0)
                if round_scale:
                    # s 是正的 normal FP32。尾数非零时进位，再清零尾数。
                    bits = T.reinterpret(s, T.uint32)
                    rounded = (bits + T.uint32(0x7FFFFF)) & T.uint32(0xFF800000)
                    s = T.reinterpret(rounded, T.float32)
                scale[i] = s
                if gy * tile_rows + i < rows:
                    scales[gy * tile_rows + i, gx] = s
            for i, j in T.Parallel(tile_rows, group_size):
                if gy * tile_rows + i < rows:
                    y[gy * tile_rows + i, gx * group_size + j] = T.clamp(
                        values[i, j] / scale[i], -448.0, 448.0
                    )

    return kernel


def my_fp8_quant(
    x: torch.Tensor, group_size: int = 128, round_scale: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """沿最后一维对有限的 BF16/FP32 输入分组，返回 E4M3FN 数据和逐组 FP32 scale。"""
    assert x.is_cuda and x.dtype in (torch.bfloat16, torch.float32)
    assert x.ndim >= 1 and group_size in (64, 128)
    width = x.shape[-1]
    assert width % group_size == 0
    y = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = x.new_empty((*x.shape[:-1], width // group_size), dtype=torch.float32)
    if x.numel():
        dtype = "bfloat16" if x.dtype == torch.bfloat16 else "float32"
        with torch.cuda.device(x.device):
            kernel = _quant_kernel(width, group_size, dtype, round_scale)
            kernel(
                x.reshape(-1, width).contiguous(),
                y.view(-1, width),
                scales.view(-1, width // group_size),
            )
    return y, scales
