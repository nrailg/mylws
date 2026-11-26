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
    packed_weight = torch.cat(txt_weights) if packed else None
    packed_bias = torch.cat(txt_biases) if packed else None
    packed_versions = [param._version for param in (*txt_weights, *txt_biases)] if packed else []

    qkv = kernels.project_joint_qkv(
        img, txt, img_weights, img_biases, txt_weights, txt_biases, packed_weight, packed_bias, packed_versions
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
    rope = model.prepare_inference_rope([[(1, 4, 4)]], txt.size(1))
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
    rope = model.prepare_inference_rope([[(1, 4, 4)]], txt.shape[1])

    actual = model(img, txt, mask, step, image_rotary_emb=rope, return_dict=False)[0]
    expected = torch.cat([
        model(
            img[i:i + 1], txt[i:i + 1], None if mask is None else mask[i:i + 1], step[i:i + 1],
            image_rotary_emb=rope, return_dict=False,
        )[0]
        for i in range(batch_size)
    ])
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("batch_size", [1, 4])
@torch.no_grad()
def test_prepared_group_text(batch_size: int) -> None:
    # Check text reuse across distinct latents, CFG prompts, and timesteps.
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    model = QwenImageTransformer2DModel(
        num_layers=2, attention_head_dim=32, num_attention_heads=2,
        joint_attention_dim=64, axes_dims_rope=(8, 12, 12),
    ).to(device, dtype=dtype).eval()
    model.enable_inference_optimizations()
    pos_txt = torch.randn(1, 8, 64, device=device, dtype=dtype)
    neg_txt = torch.randn_like(pos_txt)
    pos_mask = torch.tensor([[1, 0, 1, 1, 0, 0, 1, 1]], device=device)
    neg_mask = torch.tensor([[True, True, False, True, True, False, True, False]], device=device)
    rope = model.prepare_inference_rope([[(1, 4, 4)]], pos_txt.shape[1])
    pos = model.prepare_inference_text(pos_txt, pos_mask)
    neg = model.prepare_inference_text(neg_txt, neg_mask)
    tol = 0.0 if batch_size == 1 else 0.02

    for step_value in (0.5, 0.8):
        step = torch.tensor([step_value], device=device, dtype=dtype)
        pos_at_step = model.prepare_inference_first_block_text(pos, step, rope)
        neg_at_step = model.prepare_inference_first_block_text(neg, step, rope)
        cached_tensors = (
            pos.hidden_states, neg.hidden_states, pos.mask, neg.mask,
            *pos_at_step.first_block_qkv, *neg_at_step.first_block_qkv,
        )
        snapshots = [tensor.clone() for tensor in cached_tensors]

        for _ in range(2):
            img = torch.randn(batch_size, 16, 64, device=device, dtype=dtype)
            reference = [
                model.forward_cfg(
                    sample.unsqueeze(0), pos_txt, neg_txt, step, image_rotary_emb=rope,
                    encoder_hidden_states_mask=pos_mask, negative_encoder_hidden_states_mask=neg_mask,
                )
                for sample in img
            ]
            expected = tuple(torch.cat(branch) for branch in zip(*reference))
            projected_only = model.forward_cfg(img, pos, neg, step, image_rotary_emb=rope)
            cached = model.forward_cfg(img, pos_at_step, neg_at_step, step, image_rotary_emb=rope)
            for actual in (projected_only, cached):
                for output, expected_branch in zip(actual, expected):
                    torch.testing.assert_close(output, expected_branch, rtol=tol, atol=tol)

        for tensor, snapshot in zip(cached_tensors, snapshots):
            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
