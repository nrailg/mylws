import pytest
import torch
from tile_kernels.quant import per_block_cast

from my_fp8_quant import my_fp8_quant


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("round_scale", [False, True])
def test_partial_blocks(dtype: torch.dtype, round_scale: bool) -> None:
    """对照 PyTorch 逐块检查非连续输入、两个方向的尾块和独立的 batch scale。"""
    rng = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn((2, 259, 135), device="cuda", generator=rng, dtype=dtype).transpose(1, 2)
    y, scales = my_fp8_quant(x, round_scale=round_scale)
    assert y.shape == x.shape and y.dtype == torch.float8_e4m3fn
    assert scales.shape == (2, 2, 3) and scales.dtype == torch.float32
    for i in range(2):
        for j in range(3):
            tile = (..., slice(i * 128, (i + 1) * 128), slice(j * 128, (j + 1) * 128))
            block = x[tile].float()
            expected_s = block.abs().amax(dim=(-2, -1)).clamp_min(1e-4) / 448
            if round_scale:
                expected_s = torch.exp2(torch.ceil(torch.log2(expected_s)))
            torch.testing.assert_close(scales[:, i, j], expected_s, rtol=2e-7, atol=0)
            expected = (block / expected_s[:, None, None]).clamp(-448, 448).to(y.dtype)
            assert torch.equal(y[tile].view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("round_scale", [False, True])
def test_deepseek(round_scale: bool) -> None:
    """用块内单点峰值验证整块共享 scale，并与 DeepSeek per_block_cast 对照。"""
    peaks = torch.tensor([[0, 1e-6, 446], [448, 450, 896]], dtype=torch.bfloat16, device="cuda")
    x = peaks.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1) * 0.25
    x[127::128, 127::128] = -peaks
    y, scales = my_fp8_quant(x, round_scale=round_scale)
    ref_y, ref_s = per_block_cast(x, "e4m3", (128, 128), round_sf=round_scale)
    assert scales.shape == (2, 3)
    torch.testing.assert_close(scales, ref_s, rtol=0 if round_scale else 2e-7, atol=0)
    assert torch.equal(y.view(torch.uint8), ref_y.view(torch.uint8))


@pytest.mark.parametrize("shape", [(0, 129), (3, 0), (0, 129, 1)])
def test_empty(shape: tuple[int, ...]) -> None:
    """空输入保留前导维度及二维 block 的 scale 布局。"""
    y, scales = my_fp8_quant(torch.empty(shape, device="cuda"))
    assert y.shape == shape and y.dtype == torch.float8_e4m3fn
    assert scales.shape == (*shape[:-2], (shape[-2] + 127) // 128, (shape[-1] + 127) // 128)


def test_vector() -> None:
    """一维输入缺少组成二维 block 的行轴。"""
    with pytest.raises(AssertionError):
        my_fp8_quant(torch.empty(128, device="cuda"))
