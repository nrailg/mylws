from pathlib import Path

import torch
from transformers import __version__ as transformers_version
from transformers.models.qwen4_exp import modeling_qwen4_exp as upstream_qwen

if __package__:
    from . import modeling_qwen4_exp as local_qwen
else:
    import modeling_qwen4_exp as local_qwen


@torch.inference_mode()
def main() -> None:
    local_file = Path(__file__).with_name("modeling_qwen4_exp.py")
    assert local_file.samefile(local_qwen.__file__)
    assert not local_file.samefile(upstream_qwen.__file__)

    torch.manual_seed(0)
    settings = dict(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["indexed_attention"],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=4,
        indexer_compress_ratio=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=48,
        shared_expert_intermediate_size=32,
        hc_lowrank=16,
        attn_implementation="eager",
        experts_implementation="eager",
        use_cache=False,
    )
    upstream_config = upstream_qwen.Qwen4ExpTextConfig(**settings)
    local_config = local_qwen.Qwen4ExpTextConfig(**settings)
    reference = upstream_qwen.Qwen4ExpForCausalLM(upstream_config).to("cuda", torch.bfloat16).eval()
    model = local_qwen.Qwen4ExpForCausalLM(local_config).to("cuda", torch.bfloat16).eval()
    model.load_state_dict(reference.state_dict(), strict=True)

    input_ids = torch.arange(16, device="cuda").reshape(2, 8)
    logits = model(input_ids=input_ids).logits
    expected = reference(input_ids=input_ids).logits

    assert logits.shape == expected.shape == (*input_ids.shape, local_config.vocab_size)
    assert logits.dtype == expected.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()
    # 将 BF16 的原始位模式看作 int16，逐个 logit 比较，包括零的符号位。
    local_bits = logits.contiguous().view(torch.int16)
    upstream_bits = expected.contiguous().view(torch.int16)
    mismatches = torch.count_nonzero(local_bits != upstream_bits).item()
    assert mismatches == 0, f"{mismatches}/{logits.numel()} BF16 logits differ"
    print(f"Transformers {transformers_version}, logits {tuple(logits.shape)} on {logits.device}: "
          f"{mismatches}/{logits.numel()} BF16 bit patterns differ")


if __name__ == "__main__":
    main()
