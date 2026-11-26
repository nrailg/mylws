import pytest
import torch
import torch.nn.functional as F

from my_qwen_image import inference_kernels as kernels
from my_qwen_image.transformer_qwenimage import (
    QwenImageTransformer2DModel,
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


@pytest.mark.parametrize("packed", [False, True])
def test_joint_projection(packed: bool) -> None:
    # Check direct GEMM writes and the order of the packed text projection in the joint buffer.
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    width, img_len, txt_len = 128, 19, 7
    img = torch.randn(img_len, width, device=device, dtype=dtype)
    txt = torch.randn(txt_len, width, device=device, dtype=dtype)
    img_weights = [torch.randn(width, width, device=device, dtype=dtype) for _ in range(3)]
    img_biases = [torch.randn(width, device=device, dtype=dtype) for _ in range(3)]
    txt_weights = [torch.randn_like(img_weights[0]) for _ in range(3)]
    txt_biases = [torch.randn_like(img_biases[0]) for _ in range(3)]

    qkv = kernels.project_joint_qkv(
        img, txt, img_weights, img_biases, txt_weights, txt_biases, pack_text=packed
    )
    for actual, iw, ib, tw, tb in zip(qkv, img_weights, img_biases, txt_weights, txt_biases):
        expected = torch.cat([F.linear(txt, tw, tb), F.linear(img, iw, ib)])
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_compiled_masked_forward_after_weight_load(batch_size: int) -> None:
    # Compiled masked forward uses the text value bias updated by load_state_dict.
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    model = QwenImageTransformer2DModel(
        num_layers=1, attention_head_dim=32, num_attention_heads=2,
        joint_attention_dim=64, axes_dims_rope=(8, 12, 12),
    ).to(device, dtype=dtype).eval()
    model.enable_inference_optimizations()
    img = torch.randn(batch_size, 16, 64, device=device, dtype=dtype)
    txt = torch.randn(batch_size, 8, 64, device=device, dtype=dtype)
    mask = torch.tensor([[True, False, True, True, False, False, True, True]], device=device)
    mask = mask.expand(batch_size, -1)
    step = torch.full((batch_size,), 0.5, device=device, dtype=dtype)
    rope = model.pos_embed([[(1, 4, 4)]], max_txt_seq_len=txt.size(1), device=img.device)
    kwargs = dict(
        encoder_hidden_states_mask=mask, timestep=step, image_rotary_emb=rope, return_dict=False
    )

    compiled = torch.compile(model, dynamic=False, options={"triton.cudagraphs": False})
    state = model.state_dict()
    bias_name = "transformer_blocks.0.attn.add_v_proj.bias"
    state[bias_name] = torch.full_like(state[bias_name], 20)
    with torch.no_grad():
        before = compiled(img, txt, **kwargs)[0]
        model.load_state_dict(state)
        actual = compiled(img, txt, **kwargs)[0]
        expected = model(img, txt, **kwargs)[0]

    assert not torch.allclose(before, expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@torch.no_grad()
def test_joint_attention_rejects_mismatched_batches() -> None:
    """验证 joint attention 会在处理 QKV 前拒绝不同的 image 和 text batch size。"""
    device, dtype = "cuda", torch.bfloat16
    model = QwenImageTransformer2DModel(
        num_layers=1, attention_head_dim=32, num_attention_heads=2,
        joint_attention_dim=64, axes_dims_rope=(8, 12, 12),
    ).to(device, dtype=dtype).eval()
    model.enable_inference_optimizations()
    image = torch.zeros(2, 16, 64, device=device, dtype=dtype)
    text = torch.zeros(1, 8, 64, device=device, dtype=dtype)
    timestep = torch.full((2,), 0.5, device=device, dtype=dtype)
    with pytest.raises(AssertionError, match="Image and text batch sizes must match"):
        model(image, text, timestep=timestep, img_shapes=[[(1, 4, 4)]])


@pytest.mark.parametrize("batch_size, masked", [(2, False), (4, True)])
@torch.no_grad()
def test_batched_forward_matches_individual_samples(batch_size: int, masked: bool) -> None:
    # Check that the joint layout keeps distinct timesteps, prompts, and image rows in each sample.
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    model = QwenImageTransformer2DModel(
        num_layers=2, attention_head_dim=32, num_attention_heads=2,
        joint_attention_dim=64, axes_dims_rope=(8, 12, 12),
    ).to(device, dtype=dtype).eval()
    model.enable_inference_optimizations()
    img = torch.randn(batch_size, 16, 64, device=device, dtype=dtype)
    txt = torch.randn(batch_size, 8, 64, device=device, dtype=dtype)
    mask = torch.rand(batch_size, 8, device=device) > 0.4 if masked else None
    step = torch.linspace(0.3, 0.7, batch_size, device=device, dtype=dtype)
    rope = model.pos_embed([[(1, 4, 4)]], max_txt_seq_len=txt.shape[1], device=img.device)

    actual = model(img, txt, mask, step, image_rotary_emb=rope, return_dict=False)[0]
    expected = torch.cat([
        model(
            img[i:i + 1], txt[i:i + 1], None if mask is None else mask[i:i + 1], step[i:i + 1],
            image_rotary_emb=rope, return_dict=False,
        )[0]
        for i in range(batch_size)
    ])
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@torch.no_grad()
def test_serial_cfg_matches_independent_forwards() -> None:
    """验证不同 text length 的串行 CFG 复用 image projection 后与独立 forward 逐位一致。"""
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    model = QwenImageTransformer2DModel(
        num_layers=2, attention_head_dim=32, num_attention_heads=2,
        joint_attention_dim=64, axes_dims_rope=(8, 12, 12),
    ).to(device, dtype=dtype).eval()
    model.enable_inference_optimizations()
    image = torch.randn(1, 16, 64, device=device, dtype=dtype)
    positive = torch.randn(1, 8, 64, device=device, dtype=dtype)
    negative = torch.randn(1, 5, 64, device=device, dtype=dtype)
    negative_mask = torch.tensor([[True, False, True, True, False]], device=device)
    timestep = torch.tensor([0.5], device=device, dtype=dtype)
    img_shapes = [[(1, 4, 4)]]

    actual_positive, actual_negative = model.forward_cfg(
        image, positive, negative, timestep, img_shapes=img_shapes,
        negative_encoder_hidden_states_mask=negative_mask,
    )
    expected_positive = model(image, positive, timestep=timestep, img_shapes=img_shapes).sample
    expected_negative = model(
        image, negative, negative_mask, timestep, img_shapes=img_shapes,
    ).sample
    torch.testing.assert_close(actual_positive, expected_positive, rtol=0, atol=0)
    torch.testing.assert_close(actual_negative, expected_negative, rtol=0, atol=0)
