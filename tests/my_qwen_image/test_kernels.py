import pytest
import torch
import torch.nn.functional as F

from my_qwen_image import inference_kernels as kernels


@pytest.mark.parametrize("mod_batch", [1, 2])
def test_fused_modulation(mod_batch: int) -> None:
    # Check the batch stride of split modulation and broadcast of a single timestep.
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    width, eps = 3072, 1e-6
    x = torch.randn(2, 5, width, device=device, dtype=dtype)
    update = torch.randn_like(x)
    mod = torch.randn(mod_batch, 6 * width, device=device, dtype=dtype)
    mod1, mod2 = mod.chunk(2, dim=-1)
    shift1, scale1, gate1 = mod1.unsqueeze(1).chunk(3, dim=-1)
    shift2, scale2, gate2 = mod2.unsqueeze(1).chunk(3, dim=-1)

    expected_first = F.layer_norm(x, (width,), eps=eps) * (1 + scale1) + shift1
    expected_residual = x + gate1 * update
    expected_second = F.layer_norm(expected_residual, (width,), eps=eps) * (1 + scale2) + shift2
    expected_out = expected_residual + gate2 * update

    first = kernels.norm_modulate(x, mod1, eps)
    residual, second = kernels.residual_norm_modulate(x, update, mod1, mod2, eps)
    out = kernels.gated_residual(residual, update, mod2)
    torch.testing.assert_close(first, expected_first, rtol=0.008, atol=0.008)
    torch.testing.assert_close(second, expected_second, rtol=0.008, atol=0.008)
    torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
