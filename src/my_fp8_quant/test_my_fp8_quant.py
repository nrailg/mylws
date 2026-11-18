import pytest
import torch
from tile_kernels.quant import per_token_cast

from my_fp8_quant import my_fp8_quant


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("round_scale", [False, True])
def test_quant(dtype: torch.dtype, group_size: int, round_scale: bool) -> None:
    """对照 PyTorch 检查跨 tile 的非连续三维输入、逐组 scale 和 FP8 编码。"""
    width = group_size * 3
    rng = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn((3, 7, width), device="cuda", generator=rng, dtype=dtype).transpose(0, 1)
    y, scales = my_fp8_quant(x, group_size, round_scale)
    groups = x.float().reshape(7, 3, 3, group_size)
    expected_s = groups.abs().amax(dim=-1).clamp_min(1e-4) * (1.0 / 448.0)
    if round_scale:
        expected_s = torch.exp2(torch.ceil(torch.log2(expected_s)))
    expected = (groups / expected_s.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(scales, expected_s, rtol=0, atol=0)
    assert y.dtype == torch.float8_e4m3fn
    assert torch.equal(y.view(torch.uint8), expected.reshape(x.shape).view(torch.uint8))


@pytest.mark.parametrize("round_scale", [False, True])
def test_deepseek(round_scale: bool) -> None:
    """对照 DeepSeek TileKernels 检查零值、极小值和 2 的幂附近的量化结果。"""
    peaks = torch.tensor([0, 1e-6, 446, 448, 450, 896], dtype=torch.bfloat16, device="cuda")
    pattern = torch.tensor([1, -1, 0.5, -0.25], dtype=peaks.dtype, device=peaks.device)
    x = (peaks[:, None] * pattern.repeat(32)).reshape(3, 256)
    y, scales = my_fp8_quant(x, 128, round_scale)
    ref_y, ref_s = per_token_cast(x, "e4m3", 128, round_sf=round_scale)
    # act_quant 乘以预先舍入的 1/448，TileKernels 直接做除法；普通 scale 允许相差 1 ULP。
    max_ulps = 0 if round_scale else 1
    assert (scales.view(torch.int32) - ref_s.view(torch.int32)).abs().max().item() <= max_ulps
    assert torch.equal(y.view(torch.uint8), ref_y.view(torch.uint8))


@pytest.mark.parametrize("shape", [(0, 128), (3, 0), (0,)])
def test_empty(shape: tuple[int, ...]) -> None:
    """空输入保留前导维度和每组一个 scale 的布局。"""
    x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    y, scales = my_fp8_quant(x)
    assert y.shape == shape and y.dtype == torch.float8_e4m3fn
    assert scales.shape == (*shape[:-1], shape[-1] // 128) and scales.dtype == torch.float32


def test_unaligned_width() -> None:
    """最后一维不足整组时拒绝输入。"""
    with pytest.raises(AssertionError):
        my_fp8_quant(torch.empty(127, device="cuda"))
