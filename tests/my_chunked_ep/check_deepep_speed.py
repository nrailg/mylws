"""在单机多卡上比较 G-Core eager EP 与 Chunked DeepEP 的 forward + backward。"""

import argparse
import os
from collections.abc import Callable
from statistics import median
from time import perf_counter

import torch
import torch.distributed as dist
from deep_ep import Buffer
from torch import Tensor

from my_chunked_ep.chunked_deepep import chunked_deepep
from my_chunked_ep.eager_ep import eager_ep


TOKENS_PER_RANK = 4096
HIDDEN_SIZE = 2048
NUM_EXPERTS = 512
TOP_K = 10
NUM_CHUNKS = 2
COMM_SMS = 24
WARMUP = 4
SAMPLES = 15


def scale_experts(rows: Tensor, expert_ids: Tensor, scales: Tensor) -> Tensor:
    return rows * scales[expert_ids, None].to(rows.dtype)


def prepare_experts(ids: Tensor, counts: list[int]) -> tuple[Tensor, Tensor]:
    flat_ids = ids.flatten()
    valid = torch.nonzero_static(flat_ids >= 0, size=sum(counts)).flatten()
    return valid, flat_ids[valid]


def prepare_sorted_experts(ids: Tensor, counts: list[int]) -> tuple[Tensor, Tensor]:
    slots, expert_ids = prepare_experts(ids, counts)
    order = expert_ids.argsort(stable=True)
    return slots[order], expert_ids[order]


def compute_experts(
    rows: Tensor,
    layout: tuple[Tensor, Tensor],
    weights: Tensor,
    scales: Tensor,
) -> Tensor:
    slots, expert_ids = layout
    row_ids = slots // weights.size(1)
    projected = scale_experts(rows[row_ids], expert_ids, scales)
    weighted = (projected * weights.flatten()[slots, None]).to(rows.dtype)
    return torch.zeros_like(rows).index_add_(0, row_ids, weighted)


def make_buffer(group: dist.ProcessGroup, hidden_size: int, comm_sms: int = COMM_SMS) -> Buffer:
    size = group.size()
    Buffer.set_num_sms(comm_sms)
    configs = (Buffer.get_dispatch_config(size), Buffer.get_combine_config(size))
    nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_size * 2, size) for config in configs)
    return Buffer(group, nvl_bytes, 0, explicitly_destroy=True)


def compare(
    group: dist.ProcessGroup,
    buffer: Buffer,
    *,
    tokens_per_rank: int = TOKENS_PER_RANK,
    prepare_fn: Callable[[Tensor, list[int]], tuple[Tensor, Tensor]] = prepare_experts,
    expert_fn: Callable[[Tensor, tuple[Tensor, Tensor], Tensor, Tensor], Tensor] = compute_experts,
) -> None:
    torch.manual_seed(2026 + group.rank())
    scales = (torch.rand(NUM_EXPERTS // group.size(), device="cuda") + 0.5).requires_grad_()
    hidden = torch.randn(
        tokens_per_rank, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    ids = torch.rand(tokens_per_rank, NUM_EXPERTS, device="cuda").topk(TOP_K, dim=-1).indices
    logits = torch.randn(ids.shape, device="cuda")
    weights = logits.softmax(dim=-1).requires_grad_()
    grad_output = torch.randn_like(hidden)
    grad_inputs = (hidden, weights, scales)

    def eager() -> Tensor:
        return eager_ep(hidden, ids, weights, (scales,), scale_experts, NUM_EXPERTS, group)

    def chunked() -> Tensor:
        return chunked_deepep(
            hidden, ids, weights, (scales,), expert_fn, NUM_EXPERTS, buffer,
            prepare_fn=prepare_fn, num_chunks=NUM_CHUNKS,
        )

    def train_step(forward: Callable[[], Tensor]) -> tuple[Tensor, ...]:
        output = forward()
        gradients = torch.autograd.grad(output, grad_inputs, grad_output)
        return (output.detach(), *gradients)

    names = ("output", "hidden_grad", "routing_grad", "expert_scale_grad")
    reference = train_step(eager)
    actual = train_step(chunked)
    errors = []
    for expected, result in zip(reference, actual, strict=True):
        expected = expected.float()
        errors.append((result.float() - expected).norm() / expected.norm())
    del reference, actual, expected, result
    error_by_tensor = torch.stack(errors)
    dist.all_reduce(error_by_tensor, op=dist.ReduceOp.MAX, group=group)
    errors = dict(zip(names, error_by_tensor.tolist(), strict=True))
    assert all(error < 0.03 for error in errors.values()), errors

    backends = (eager, chunked)
    samples = ([], [])
    for iteration in range(WARMUP + SAMPLES):
        order = (0, 1) if iteration % 2 == 0 else (1, 0)
        for index in order:
            dist.barrier(group=group)
            torch.cuda.synchronize()
            start = perf_counter()
            train_step(backends[index])
            torch.cuda.synchronize()
            if iteration >= WARMUP:
                samples[index].append(1000 * (perf_counter() - start))

    times = torch.tensor(samples, dtype=torch.float64, device="cuda")
    dist.all_reduce(times, op=dist.ReduceOp.MAX, group=group)
    eager_samples, chunked_samples = times.tolist()
    eager_ms, chunked_ms = median(eager_samples), median(chunked_samples)
    if group.rank() == 0:
        print(f"GPU={torch.cuda.get_device_name()} EP={group.size()} "
              f"tokens/rank={tokens_per_rank} hidden={HIDDEN_SIZE} "
              f"experts={NUM_EXPERTS} top-k={TOP_K} chunks={NUM_CHUNKS}")
        print(f"relative L2 errors: {errors}")
        print(f"gcore_eager samples (ms): {eager_samples}")
        print(f"chunked_deepep samples (ms): {chunked_samples}")
        print(f"gcore_eager={eager_ms:.3f} ms chunked_deepep={chunked_ms:.3f} ms "
              f"speedup={eager_ms / chunked_ms:.3f}x")
    assert chunked_ms < eager_ms, f"chunked_deepep {chunked_ms:.3f} ms >= eager {eager_ms:.3f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-slots", action="store_true", help="使用 G-Core fused slot 算子")
    parser.add_argument("--comm-sms", type=int, default=COMM_SMS, help="DeepEP 通信使用的 SM 数")
    parser.add_argument("--tokens-per-rank", type=int, default=TOKENS_PER_RANK)
    args = parser.parse_args()
    assert args.tokens_per_rank > 0
    assert args.comm_sms > 0 and args.comm_sms % 2 == 0

    prepare_fn, expert_fn = prepare_experts, compute_experts
    if args.fused_slots:
        from my_chunked_ep.moe_slots import gather_slots, weighted_reduce_slots

        def compute_fused_experts(
            rows: Tensor, layout: tuple[Tensor, Tensor], weights: Tensor, scales: Tensor,
        ) -> Tensor:
            slots, expert_ids = layout
            expanded, inverse = gather_slots(rows, slots, weights.size(1))
            projected = scale_experts(expanded, expert_ids, scales)
            return weighted_reduce_slots(projected, weights, slots, inverse)

        prepare_fn, expert_fn = prepare_sorted_experts, compute_fused_experts

    world = int(os.environ["WORLD_SIZE"])
    assert world == int(os.environ["LOCAL_WORLD_SIZE"]) and world in (2, 4, 8)
    assert torch.cuda.device_count() >= world
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        group = dist.group.WORLD
        if group.rank() == 0:
            print(f"comm_sms={args.comm_sms} fused_slots={args.fused_slots}")
        buffer = make_buffer(group, HIDDEN_SIZE, comm_sms=args.comm_sms)
        try:
            compare(
                group, buffer, tokens_per_rank=args.tokens_per_rank, prepare_fn=prepare_fn,
                expert_fn=expert_fn,
            )
        finally:
            buffer.destroy()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
