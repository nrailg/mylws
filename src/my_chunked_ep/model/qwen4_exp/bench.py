import argparse
import gc
from pathlib import Path
from time import perf_counter

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
    model_class: type[PreTrainedModel],
    checkpoint: Path,
    device_map: str | dict,
    debug_num_layers: int | None = None,
) -> PreTrainedModel:
    config = None
    skipped_layers = ()
    if debug_num_layers is not None:
        config = model_class.config_class.from_pretrained(checkpoint, local_files_only=True)
        depth = config.num_hidden_layers
        assert 1 <= debug_num_layers <= depth, f"debug_num_layers must be between 1 and {depth}"
        skipped_layers = tuple(f"model.layers.{i}." for i in range(debug_num_layers, depth))
        config.num_hidden_layers = debug_num_layers
        config.layer_types = config.layer_types[:debug_num_layers]
        # PLE 的层号从 1 开始。截掉其所在层时，loader 会跳过整张 n-gram embedding。
        config.ple_layer_ids = [i for i in config.ple_layer_ids if i <= debug_num_layers]
        if not config.ple_layer_ids:
            config.number_of_conv_states = 1
    model, info = model_class.from_pretrained(
        checkpoint,
        config=config,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="eager",
        experts_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    info["unexpected_keys"] = {
        key for key in info["unexpected_keys"] if not key.startswith(skipped_layers)
    }
    assert not any(info.values()), info
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    return model.eval()


@torch.inference_mode()
def main(checkpoint: Path, debug_num_layers: int | None = None) -> None:
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
    total_load_seconds = 0.0
    for model_class in (LocalQwen4ExpForCausalLM, pub_qwen.Qwen4ExpForCausalLM):
        print(f"Loading {checkpoint.name} with {model_class.__name__}", flush=True)
        start = perf_counter()
        model = load_checkpoint(model_class, checkpoint, device_map, debug_num_layers)
        # 等待各张 GPU 完成权重传输，计时在 forward 前结束。
        for device in {p.device for p in model.parameters() if p.is_cuda}:
            torch.cuda.synchronize(device)
        load_seconds = perf_counter() - start
        total_load_seconds += load_seconds
        print(f"{model_class.__name__} load_checkpoint: {load_seconds:.2f} s", flush=True)

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

    print(f"Total load_checkpoint: {total_load_seconds:.2f} s")
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
        "checkpoint",
        nargs="?",
        type=Path,
        default=Path("/hf-hub/Qwen/Qwen3.8-Flash-Next"),
        help="Local checkpoint directory (default: %(default)s)",
    )
    parser.add_argument(
        "--debug-num-layers", "--debug-truncate-num-hidden-layers",
        type=int, metavar="N",
        help="Load the first N decoder layers (default: all checkpoint layers)",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.debug_num_layers)
