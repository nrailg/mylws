import argparse
import gc
from pathlib import Path

import torch
from transformers import PreTrainedModel
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen4_exp import modeling_qwen4_exp as pub_qwen

if __package__:
    from . import modeling_qwen4_exp as local_qwen
else:  # 直接执行 bench.py 时，Python 从脚本所在目录导入。
    import modeling_qwen4_exp as local_qwen


class LocalQwen4ExpForCausalLM(local_qwen.Qwen4ExpForCausalLM):
    """用单独的类名为本地实现注册 checkpoint 转换规则。"""


def load_checkpoint(
    model_class: type[PreTrainedModel], checkpoint: Path, device_map: str | dict
) -> PreTrainedModel:
    model, info = model_class.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="eager",
        experts_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    assert not any(info.values()), info
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    return model.eval()


@torch.inference_mode()
def main(checkpoint: Path) -> None:
    assert checkpoint.is_dir(), checkpoint
    local_file = Path(__file__).with_name("modeling_qwen4_exp.py")
    assert local_file.samefile(local_qwen.__file__)
    assert not local_file.samefile(pub_qwen.__file__)
    assert torch.cuda.device_count() > 0

    # Transformers 要求本地类显式注册权重前缀转换和 PLE embedding 分片拼接规则。
    register_checkpoint_conversion_mapping(
        LocalQwen4ExpForCausalLM.__name__,
        get_checkpoint_conversion_mapping(local_qwen.Qwen4ExpTextConfig.model_type),
        overwrite=True,
    )
    input_ids = torch.arange(16).reshape(2, 8)
    outputs = []
    device_map = "auto"
    for model_class in (LocalQwen4ExpForCausalLM, pub_qwen.Qwen4ExpForCausalLM):
        print(f"Loading {checkpoint.name} with {model_class.__name__}", flush=True)
        model = load_checkpoint(model_class, checkpoint, device_map)
        # 复用第一份模型的设备分布，两次前向只保留 CPU 上的 logits。
        device_map = model.hf_device_map
        device = model.get_input_embeddings().weight.device
        logits = model(input_ids=input_ids.to(device), use_cache=False).logits.cpu()
        assert logits.shape == (*input_ids.shape, model.config.vocab_size)
        assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
        outputs.append(logits)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    logits, expected = outputs
    assert logits.shape == expected.shape
    # 将 BF16 的原始位模式看作 int16，逐个 logit 比较，包括零的符号位。
    local_bits = logits.contiguous().view(torch.int16)
    pub_bits = expected.contiguous().view(torch.int16)
    mismatches = torch.count_nonzero(local_bits != pub_bits).item()
    assert mismatches == 0, f"{mismatches}/{logits.numel()} BF16 logits differ"
    print(
        f"{checkpoint.name}, logits {tuple(logits.shape)}: "
        f"{mismatches}/{logits.numel()} BF16 bit patterns differ"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint", type=Path, help="Local Qwen4-Exp Hugging Face checkpoint directory"
    )
    main(parser.parse_args().checkpoint)
