"""在 meta 上划分 Qwen4-Exp，再从 HF safetensors 读取每个 rank 的权重。"""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard
from transformers.modeling_utils import local_torch_dtype

if __package__:
    from . import modeling_qwen4_exp as qwen
    from .parallel_layers import (
        ParallelExperts, RowEmbedding, UlyssesAttention, UlyssesGatedDeltaNet, UlyssesPLE,
    )
else:
    import modeling_qwen4_exp as qwen
    from parallel_layers import (
        ParallelExperts, RowEmbedding, UlyssesAttention, UlyssesGatedDeltaNet, UlyssesPLE,
    )


@dataclass
class Parallelism:
    dense: DeviceMesh
    ep: DeviceMesh
    expert_fsdp: DeviceMesh
    cp: DeviceMesh
    fsdp: bool

    @classmethod
    def create(cls, fsdp: bool, ep_size: int, cp_size: int) -> "Parallelism":
        world = dist.get_world_size()
        assert ep_size >= 1 and world % ep_size == 0
        assert cp_size >= 1 and world % cp_size == 0
        dense = init_device_mesh("cuda", (world,))
        experts = init_device_mesh(
            "cuda", (world // ep_size, ep_size), mesh_dim_names=("expert_fsdp", "ep")
        )
        sequences = init_device_mesh(
            "cuda", (world // cp_size, cp_size), mesh_dim_names=("data", "cp")
        )
        return cls(dense, experts["ep"], experts["expert_fsdp"], sequences["cp"], fsdp)


def local_tensor(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@torch.no_grad()
def load_local_weights(
    model: nn.Module, checkpoint: Path, shapes: dict[str, torch.Size],
    starts: dict[str, int], ple_parts: int,
) -> None:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]

    def read(key: str, shape: tuple[int, ...], first: int, out: Tensor) -> None:
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as handle:
            source = handle.get_slice(key)
            assert tuple(source.get_shape()) == tuple(shape), key
            if out.numel():
                out.copy_(source[first:first + out.size(0)])

    for name, param in model.state_dict(keep_vars=True).items():
        shape = shapes[name]
        out = local_tensor(param)
        first = starts.get(name, 0)
        if isinstance(param, DTensor):
            assert param.placements == (Shard(0),)
            mesh = param.device_mesh
            rows, offset = Shard.local_shard_size_and_offset(
                param.size(0), mesh.size(), mesh.get_local_rank()
            )
            assert rows == out.size(0), name
            first += offset
        # 发布版把文本子模型嵌在 language_model 下。普通文本 checkpoint 可以直接使用原名。
        key = name if name in index else name.replace("model.", "model.language_model.", 1)
        if key in index:
            read(key, shape, first, out)
            continue
        assert name.endswith("ngram_embedding.weight"), f"Missing checkpoint weight: {name}"
        prefix = key.removesuffix(".weight")
        if f"{prefix}.shard_0.weight" not in index:
            prefix = name.removesuffix(".weight")
        assert shape[0] % ple_parts == 0
        part_rows = shape[0] // ple_parts
        stop = first + out.size(0)
        for part in range(first // part_rows, (stop + part_rows - 1) // part_rows):
            base = part * part_rows
            lo, hi = max(first, base), min(stop, base + part_rows)
            read(f"{prefix}.shard_{part}.weight", (part_rows, *shape[1:]),
                 lo - base, out[lo - first:hi - first])


def build_parallel_model(
    config: qwen.Qwen4ExpTextConfig, checkpoint: Path, parallel: Parallelism,
    dtype: torch.dtype = torch.bfloat16,
) -> qwen.Qwen4ExpForCausalLM:
    config._attn_implementation = config._experts_implementation = "eager"
    config.dtype = dtype
    cp = parallel.cp.size()
    if cp > 1:
        assert config.num_attention_heads % cp == 0
        assert config.linear_num_key_heads % cp == 0 and config.linear_num_value_heads % cp == 0
    starts = {}
    ignored = set()
    with init_empty_weights(include_buffers=False), local_torch_dtype(dtype):
        model = qwen.Qwen4ExpForCausalLM(config)
        shapes = {name: value.shape for name, value in model.state_dict().items()}
        for idx, layer in enumerate(model.model.layers):
            prefix = f"model.layers.{idx}."
            if cp > 1:
                if layer.layer_type == "linear_attention":
                    layer.linear_attn = UlyssesGatedDeltaNet(config, idx)
                    layer.linear_attn.group = parallel.cp.get_group()
                else:
                    layer.self_attn = UlyssesAttention(config, idx)
                    layer.self_attn.group = parallel.cp.get_group()
            if parallel.ep.size() > 1:
                experts = ParallelExperts(config, parallel.ep.get_group())
                layer.mlp.experts = experts
                for name in ("gate_up_proj", "down_proj"):
                    starts[prefix + "mlp.experts." + name] = experts.start
            if layer.ple is not None:
                if cp > 1:
                    layer.ple = UlyssesPLE(config, idx, config.ple_layer_ids.index(idx + 1))
                    layer.ple.group = parallel.cp.get_group()
                ple = layer.ple.ple_embedding
                table = RowEmbedding(
                    *ple.ngram_embedding.weight.shape, parallel.dense.get_group(), dtype
                )
                ple.ngram_embedding = table
                starts[prefix + "ple.ple_embedding.ngram_embedding.weight"] = table.start
                ignored.add(table.weight)

    if parallel.fsdp:
        for layer in model.model.layers:
            fully_shard(layer.mlp.experts, mesh=parallel.expert_fsdp, reshard_after_forward=True)
            fully_shard(
                layer, mesh=parallel.dense, ignored_params=ignored, reshard_after_forward=True
            )
        fully_shard(model, mesh=parallel.dense, ignored_params=ignored, reshard_after_forward=True)

    # to_empty 也会清空 rotary 等非持久 buffer，保留构造时计算的值。
    buffers = dict(model.named_buffers())
    model.to_empty(device=torch.cuda.current_device())
    for name, buffer in model.named_buffers():
        buffer.copy_(buffers[name])
    load_local_weights(model, checkpoint, shapes, starts, config.split_ngram_parts)
    return model


def sequence_logits(model: nn.Module, input_ids: Tensor, parallel: Parallelism) -> Tensor:
    """输入为 CP 组共享的等长序列，返回当前 rank 连续 token 段的 logits。"""
    cp = parallel.cp.size()
    if cp == 1:
        return model(input_ids=input_ids, use_cache=False).logits
    length = input_ids.size(1)
    assert length % cp == 0 and length > 0, "Ulysses needs equally sized nonempty sequence shards"
    rank = parallel.cp.get_local_rank()
    local_ids = input_ids.chunk(cp, dim=1)[rank].contiguous()
    positions = torch.arange(length, device=input_ids.device).chunk(cp)[rank][None, :]
    positions = positions.expand(input_ids.size(0), -1)
    dtype = model.lm_head.weight.dtype
    causal = torch.full(
        (length, length), torch.finfo(dtype).min, dtype=dtype, device=input_ids.device
    ).triu(1)
    masks = {
        "linear_attention": None, "indexed_attention": causal.expand(input_ids.size(0), 1, -1, -1)
    }
    return model(
        input_ids=local_ids, position_ids=positions, attention_mask=masks, use_cache=False
    ).logits


@torch.no_grad()
def finish_gradients(model: nn.Module, parallel: Parallelism) -> None:
    """将各 rank 局部 loss 的梯度归一化为 world mean，FSDP 管理的参数已做过组内 AVG。"""
    world = parallel.dense.size()
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = local_tensor(param.grad)
        if name.endswith("ngram_embedding.weight"):
            grad.div_(world)
        elif ".mlp.experts." in name and parallel.ep.size() > 1:
            if parallel.fsdp:
                grad.div_(parallel.ep.size())
            else:
                dist.all_reduce(grad, group=parallel.expert_fsdp.get_group())
                grad.div_(world)
        elif not parallel.fsdp:
            dist.all_reduce(grad, group=parallel.dense.get_group())
            grad.div_(world)
