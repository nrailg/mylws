import pytest
import torch
import torch.nn.functional as F

from my_qwen_image import inference_kernels as kernels
from my_qwen_image.transformer_qwenimage import (
    RMSNorm,
    apply_rotary_emb_qwen,
)


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


def test_fused_qk_norm_rope() -> None:
    # Rotate Q and K in place in a joint buffer view with padded rows.
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    heads, head_dim, skip, seq = 4, 128, 8, 17
    dim = heads * head_dim
    with torch.no_grad():
        joint_q = torch.randn(skip + seq, dim * 2, device=device, dtype=dtype)
        joint_k = torch.randn_like(joint_q)
        q, k = joint_q[skip:, :dim], joint_k[skip:, :dim]
        q_weight = RMSNorm(head_dim, eps=1e-6).to(device, dtype=dtype)
        k_weight = RMSNorm(head_dim, eps=3e-6).to(device, dtype=dtype)
        q_weight.weight.uniform_(0.5, 1.5)
        k_weight.weight.uniform_(0.5, 1.5)
        angle = torch.rand(seq, head_dim // 2, device=device)
        freqs = torch.polar(torch.ones_like(angle), angle)

        q_ref = q_weight(q.view(1, seq, heads, head_dim))
        k_ref = k_weight(k.view(1, seq, heads, head_dim))
        q_ref = apply_rotary_emb_qwen(q_ref, freqs, use_real=False).reshape_as(q)
        k_ref = apply_rotary_emb_qwen(k_ref, freqs, use_real=False).reshape_as(k)
        before_q, before_k = joint_q.clone(), joint_k.clone()

        kernels.qk_norm_rope_(
            q, k, q_weight.weight, k_weight.weight, torch.view_as_real(freqs), q_weight.eps, k_weight.eps
        )

    torch.testing.assert_close(q, q_ref, rtol=0.008, atol=0.008)
    torch.testing.assert_close(k, k_ref, rtol=0.008, atol=0.008)
    assert torch.equal(joint_q[:skip], before_q[:skip])
    assert torch.equal(joint_k[:skip], before_k[:skip])
    assert torch.equal(joint_q[:, dim:], before_q[:, dim:])
    assert torch.equal(joint_k[:, dim:], before_k[:, dim:])
