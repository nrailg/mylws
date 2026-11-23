import argparse
import gc
import os
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen4_exp import modeling_qwen4_exp as pub_qwen

if __package__:
    from . import modeling_qwen4_exp as local_qwen
    from .parallel_model import Parallelism, build_parallel_model, local_tensor, sequence_logits
else:  # 直接执行 bench.py 时，Python 从脚本所在目录导入。
    import modeling_qwen4_exp as local_qwen
    from parallel_model import Parallelism, build_parallel_model, local_tensor, sequence_logits


class LocalQwen4ExpForCausalLM(local_qwen.Qwen4ExpForCausalLM):
    """用单独的类名为本地实现注册 checkpoint 转换规则。"""


def read_config(
    model_class: type[PreTrainedModel], checkpoint: Path, depth: int | None,
) -> tuple[PreTrainedConfig, tuple[str, ...]]:
    config = model_class.config_class.from_pretrained(checkpoint, local_files_only=True)
    original_depth = config.num_hidden_layers
    if depth is not None:
        assert 1 <= depth <= original_depth, f"debug_num_layers must be between 1 and {original_depth}"
        config.num_hidden_layers = depth
        config.layer_types = config.layer_types[:depth]
        config.ple_layer_ids = [i for i in config.ple_layer_ids if i <= depth]
        if not config.ple_layer_ids:
            config.number_of_conv_states = 1
    skipped = tuple(f"model.layers.{i}." for i in range(config.num_hidden_layers, original_depth))
    return config, skipped


def load_checkpoint(
    model_class: type[PreTrainedModel],
    checkpoint: Path,
    device_map: str | dict,
    debug_num_layers: int | None = None,
) -> PreTrainedModel:
    config = None
    skipped_layers = ()
    if debug_num_layers is not None:
        config, skipped_layers = read_config(model_class, checkpoint, debug_num_layers)
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

        # Transformers 只在跨设备 dispatch 或磁盘 offload 时记录 hf_device_map。
        device_map = getattr(model, "hf_device_map", "auto")
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


def benchmark_tokens(sample: int, device: torch.device, vocab_size: int) -> torch.Tensor:
    return torch.arange(sample * 16, (sample + 1) * 16, device=device).reshape(2, 8) % vocab_size


def compare_reference(
    outputs: list[torch.Tensor], checkpoint: Path, depth: int | None, parallel: Parallelism,
) -> tuple[int, int, float]:
    reference = load_checkpoint(pub_qwen.Qwen4ExpForCausalLM, checkpoint, "auto", depth)
    cp = parallel.cp.size()
    mismatches, numel, max_error = 0, 0, 0.0
    for sample in range(len(outputs) // cp):
        tokens = benchmark_tokens(
            sample, reference.get_input_embeddings().weight.device, reference.config.vocab_size
        )
        expected = reference(input_ids=tokens, use_cache=False).logits.cpu()
        actual = torch.cat(outputs[sample * cp:(sample + 1) * cp], dim=1)
        different = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
        error = (actual.float() - expected.float()).abs().max().item()
        mismatches += different
        numel += expected.numel()
        max_error = max(max_error, error)
        if cp == parallel.ep.size() == 1:
            assert different == 0, (sample, different, error)
        else:
            summary = f"sample={sample} bit_mismatches={different} max_abs={error}"
            torch.testing.assert_close(
                actual, expected, rtol=0.016, atol=0.03125, msg=lambda detail: f"{summary}\n{detail}"
            )
    return mismatches, numel, max_error


@torch.no_grad()
def distributed_forward(checkpoint: Path, depth: int | None, parallel: Parallelism) -> None:
    # EP 和 CP 会改变每个 expert 的 GEMM 行数。两边使用同一 cuBLASLt 精度设置做 BF16 对拍。
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    device = torch.device("cuda", torch.cuda.current_device())
    rank, world = dist.get_rank(), dist.get_world_size()
    cp = parallel.cp.size()
    config, _ = read_config(local_qwen.Qwen4ExpForCausalLM, checkpoint, depth)
    ids = benchmark_tokens(rank // cp, device, config.vocab_size)
    assert ids.size(1) % cp == 0, "Benchmark sequence length must be divisible by CP size"
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    model = build_parallel_model(config, checkpoint, parallel).eval()
    torch.cuda.synchronize()
    load_s = perf_counter() - start
    logits = sequence_logits(model, ids, parallel).contiguous()
    torch.cuda.synchronize()
    step_s = perf_counter() - start
    assert logits.shape == (2, 8 // cp, config.vocab_size)
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
    layer = model.model.layers[0]
    ple_shapes = [
        tuple(block.ple.ple_embedding.ngram_embedding.weight.shape)
        for block in model.model.layers if block.ple is not None
    ]
    print(
        f"rank={rank} load_s={load_s:.3f} load_forward_s={step_s:.3f} "
        f"peak_GiB={torch.cuda.max_memory_allocated() / 2**30:.3f} "
        f"dense={tuple(local_tensor(layer.mlp.gate.weight).shape)} "
        f"expert={tuple(local_tensor(layer.mlp.experts.gate_up_proj).shape)} PLE={ple_shapes}",
        flush=True,
    )

    outputs = [torch.empty_like(logits) for _ in range(world)] if rank == 0 else None
    dist.gather(logits, gather_list=outputs, dst=0)
    if rank == 0:
        outputs = [x.cpu() for x in outputs]
    del model, layer, logits
    gc.collect()
    torch.cuda.empty_cache()
    # 官方 reference 可能使用任一张 GPU，其他进程在主机等待，释放 GPU 执行资源。
    host_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    dist.barrier(group=host_group)
    if rank == 0:
        mismatches, numel, max_error = compare_reference(outputs, checkpoint, depth, parallel)
        print(
            f"PASS fsdp={parallel.fsdp} ep={parallel.ep.size()} cp={cp} "
            f"layers={config.num_hidden_layers} bit_mismatches={mismatches}/{numel} "
            f"max_abs={max_error:.6g} with_reference_s={perf_counter() - start:.3f}", flush=True,
        )
    dist.barrier(group=host_group)


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
    parser.add_argument("--fsdp", action="store_true", help="Shard dense weights using FSDP2")
    parser.add_argument("--ep-size", type=int, default=1, help="Number of expert-parallel ranks")
    parser.add_argument("--cp-size", type=int, default=1, help="Number of Ulysses sequence shards")
    args = parser.parse_args()
    assert args.ep_size > 0 and args.cp_size > 0
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed or args.fsdp or args.ep_size > 1 or args.cp_size > 1:
        assert "LOCAL_RANK" in os.environ, "Launch distributed modes with torchrun"
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(
            "nccl", device_id=torch.device("cuda", torch.cuda.current_device()),
            timeout=timedelta(minutes=30),
        )
        meshes = Parallelism.create(args.fsdp, args.ep_size, args.cp_size)
        distributed_forward(args.checkpoint, args.debug_num_layers, meshes)
        # 成功时所有 rank 同步销毁。异常直接交给 torchrun，以免在集体销毁中遮住原始错误。
        dist.destroy_process_group()
    else:
        main(args.checkpoint, args.debug_num_layers)
