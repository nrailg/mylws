"""用 torchrun 在单机多卡核对四层 FP32 forward、输入梯度及各 rank 的权重梯度。"""
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard
from transformers.models.qwen4_exp import modeling_qwen4_exp as public

from my_chunked_ep.model.qwen4_exp import modeling_qwen4_exp as local
from my_chunked_ep.model.qwen4_exp.parallel_layers import ParallelExperts
from my_chunked_ep.model.qwen4_exp.parallel_model import (
    Parallelism, build_parallel_model, finish_gradients, local_tensor, sequence_logits,
)


def tiny_config() -> dict:
    return dict(
        vocab_size=96, hidden_size=32, num_hidden_layers=4, num_attention_heads=8,
        num_key_value_heads=2, head_dim=8, linear_num_key_heads=4, linear_num_value_heads=8,
        linear_key_head_dim=8, linear_value_head_dim=8, hc_count=2, hc_lowrank=8,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=24, indexer_n_heads=2, indexer_kv_heads=1,
        indexer_head_dim=8, indexer_budget=2, indexer_compress_ratio=2, eos_token_id=2,
        ple_layer_ids=[2], ple_embed_dim=16, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=31, make_ngram_vocab_size_divisible_by=16, split_ngram_parts=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        attn_implementation="eager", experts_implementation="eager",
    )


def write_fixture(root: Path, reference: torch.nn.Module) -> None:
    # 用真实 checkpoint 的前缀和 PLE 分片格式，以小 tensor 覆盖加载器与反向。
    weights = {}
    for name, value in reference.state_dict().items():
        key = name.replace("model.", "model.language_model.", 1)
        if key.endswith("ngram_embedding.weight"):
            for i, part in enumerate(value.chunk(4)):
                weights[key.removesuffix("weight") + f"shard_{i}.weight"] = part.clone()
        else:
            weights[key] = value
    filename = "weights.safetensors"
    save_file(weights, root / filename)
    index = dict.fromkeys(weights, filename)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))


def check_empty_expert(config: local.Qwen4ExpTextConfig, parallel: Parallelism) -> None:
    # EP rank 0 只接收，其余 rank 只发送，各自的输入及权重梯度都应与完整 batch 一致。
    torch.manual_seed(5)
    dense = public.Qwen4ExpTextExperts(public.Qwen4ExpTextConfig(**tiny_config())).cuda()
    for p in dense.parameters():
        torch.nn.init.normal_(p, std=0.08)
    ep = ParallelExperts(config, parallel.ep.get_group()).cuda()
    with torch.no_grad():
        for param, original in zip(ep.parameters(), dense.parameters(), strict=True):
            param.copy_(original[ep.start:ep.start + ep.num_experts])
    counts = list(range(parallel.ep.size()))
    rank = parallel.ep.get_local_rank()
    start = sum(counts[:rank])
    rows = slice(start, start + counts[rank])
    total = sum(counts)
    ref_x = torch.randn(total, config.hidden_size, device="cuda", requires_grad=True)
    ref_w = torch.tensor([[0.75, 0.25]], device="cuda").repeat(total, 1).requires_grad_()
    indices = torch.tensor([[0, 1]], device="cuda").repeat(total, 1)
    x = ref_x[rows].detach().requires_grad_()
    w = ref_w[rows].detach().requires_grad_()
    expected = dense(ref_x, indices, ref_w)
    actual = ep(x, indices[rows], w)
    torch.testing.assert_close(actual, expected[rows])
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad[rows])
    torch.testing.assert_close(w.grad, ref_w.grad[rows])
    for param, original in zip(ep.parameters(), dense.parameters(), strict=True):
        expected_grad = original.grad[ep.start:ep.start + ep.num_experts]
        torch.testing.assert_close(param.grad, expected_grad)


def check_model(root: Path, reference: torch.nn.Module, parallel: Parallelism) -> None:
    # CP=4 时 local_len=3，小于 PLE 所需历史。QSA budget=2 会真实丢弃早期块。
    config = local.Qwen4ExpTextConfig(**tiny_config())
    model = build_parallel_model(config, root, parallel, torch.float32)
    reference = reference.cuda()
    world = dist.get_world_size()
    dense = model.model.layers[0].mlp.gate.weight
    experts = model.model.layers[0].mlp.experts.gate_up_proj
    table = model.model.layers[1].ple.ple_embedding.ngram_embedding.weight
    assert isinstance(dense, DTensor) == parallel.fsdp
    assert isinstance(experts, DTensor) == parallel.fsdp
    assert local_tensor(dense).size(0) == config.num_experts // (world if parallel.fsdp else 1)
    expert_shards = world if parallel.fsdp else parallel.ep.size()
    assert local_tensor(experts).size(0) == config.num_experts // expert_shards
    full_table = reference.model.layers[1].ple.ple_embedding.ngram_embedding.weight
    assert not isinstance(table, DTensor) and table.size(0) * world == full_table.size(0)
    embeddings = []

    def retain_input(module: torch.nn.Module, args: tuple, output: Tensor) -> None:
        output.retain_grad()
        embeddings.append(output)

    reference.model.embed_tokens.register_forward_hook(retain_input)
    model.model.embed_tokens.register_forward_hook(retain_input)
    selections = []
    for network in (reference, model):
        indexer = network.model.layers[3].self_attn.indexer
        indexer.register_forward_hook(lambda module, args, output: selections.append(output))
    cp = parallel.cp.size()
    samples = world // cp
    ids = (torch.arange(samples * 24, device="cuda").reshape(-1, 12) + 3) % config.vocab_size
    ids[:, 5] = config.eos_token_id
    ref = reference(ids, use_cache=False).logits
    sample = dist.get_rank() // cp
    batch = slice(2 * sample, 2 * sample + 2)
    actual = sequence_logits(model, ids[batch], parallel)
    rank = parallel.cp.get_local_rank()
    expected = ref[batch].chunk(cp, 1)[rank]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert torch.equal(selections[1], selections[0][batch])
    selected = selections[1][:, 0, -1].sum(-1)
    assert ((selected > 0) & (selected < ids.size(1))).all()
    ref.square().sum(-1).mean().backward()
    actual.square().sum(-1).mean().backward()
    finish_gradients(model, parallel)
    torch.testing.assert_close(
        embeddings[1].grad / world, embeddings[0].grad[batch].chunk(cp, 1)[rank],
        atol=1e-5, rtol=1e-4,
    )
    original = dict(reference.named_parameters())
    for name in ("model.layers.0.mlp.experts.gate_up_proj",
                 "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
                 "model.layers.3.self_attn.q_proj.weight"):
        assert original[name].grad.count_nonzero(), name
    for name, param in model.named_parameters():
        target = original[name].grad
        if target is None:
            assert param.grad is None, name
            continue
        if ".mlp.experts." in name:
            target = target.chunk(parallel.ep.size())[parallel.ep.get_local_rank()]
        if name.endswith("ngram_embedding.weight"):
            target = target.chunk(world)[dist.get_rank()]
        if isinstance(param, DTensor):
            mesh = param.device_mesh
            count, start = Shard.local_shard_size_and_offset(target.size(0), mesh.size(), mesh.get_local_rank())
            target = target.narrow(0, start, count)
        torch.testing.assert_close(
            local_tensor(param.grad), target, rtol=1e-4, atol=1e-5, msg=lambda detail: f"{name}: {detail}"
        )


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    try:
        torch.manual_seed(2026)
        reference = public.Qwen4ExpForCausalLM(public.Qwen4ExpTextConfig(**tiny_config())).eval()
        with torch.no_grad():
            for layer in reference.model.layers:
                layer.mlp.gate.weight.normal_(std=0.1)
                if layer.ple is not None:
                    layer.ple.conv1d.weight.normal_(std=0.1)
        with TemporaryDirectory() as temp:
            paths = [temp]
            if dist.get_rank() == 0:
                write_fixture(Path(temp), reference)
            dist.broadcast_object_list(paths, src=0)
            checkpoint = Path(paths[0])
            meshes = Parallelism.create(True, 4, 4)
            for fsdp in (False, True):
                check_model(checkpoint, deepcopy(reference), replace(meshes, fsdp=fsdp))
            # expert FSDP mesh 只有一个 rank 时，梯度仍按 world size 归一化。
            full_ep = Parallelism.create(True, dist.get_world_size(), 4)
            check_model(checkpoint, deepcopy(reference), full_ep)
            check_empty_expert(local.Qwen4ExpTextConfig(**tiny_config()), meshes)
            dist.barrier()
            if dist.get_rank() == 0:
                print("PASS FP32 logits, input and parameter gradients, empty experts, PLE halo and QSA top-k")
    finally:
        dist.destroy_process_group()
