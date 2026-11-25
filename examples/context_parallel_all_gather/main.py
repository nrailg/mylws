import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from attention import AllGatherAttention


def check_correctness(group: dist.ProcessGroup, sm_margins: list[int], backend: str) -> None:
    """验证 B>1、MHA/GQA、causal/noncausal 的输出，reference 使用 FP32 SDPA math。"""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    for q_heads, kv_heads in [(4, 4), (8, 4)]:
        torch.manual_seed(123)
        q_full = torch.randn(2, 128 * world_size, q_heads, 128, device="cuda").bfloat16()
        k_full = torch.randn(2, 128 * world_size, kv_heads, 128, device="cuda").bfloat16()
        v_full = torch.randn_like(k_full)
        shard = slice(rank * 128, (rank + 1) * 128)
        q, k, v = [x[:, shard].contiguous() for x in (q_full, k_full, v_full)]
        for causal in (False, True):
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(
                    q_full.transpose(1, 2).float(),
                    k_full.transpose(1, 2).float(),
                    v_full.transpose(1, 2).float(),
                    is_causal=causal,
                    enable_gqa=True,
                ).transpose(1, 2)[:, shard]
            for heads_per_group in (1, 2, 4):
                attention = AllGatherAttention(q, k, heads_per_group, causal, group, backend)
                for margin in sm_margins:
                    serial = attention.forward(q, k, v, overlap=False, sm_margin=margin)
                    asynchronous = attention.forward(q, k, v, overlap=True, sm_margin=margin)
                    torch.testing.assert_close(serial.float(), expected, atol=0.002, rtol=0.02)
                    torch.testing.assert_close(asynchronous, serial, atol=0, rtol=0)
                torch.cuda.synchronize()
    if rank == 0:
        print("PASS: BSHD B=2, MHA/GQA, causal/noncausal, head groups, SM margins", flush=True)


def benchmark(
    attention: AllGatherAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    overlap: bool,
    margin: int,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        attention.forward(q, k, v, overlap, margin)
    torch.cuda.synchronize()
    dist.barrier()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(iterations):
        attention.forward(q, k, v, overlap, margin)
    end.record()
    end.synchronize()
    elapsed = torch.tensor(start.elapsed_time(end) / iterations, device=q.device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def main() -> None:
    parser = argparse.ArgumentParser(description="BSHD AllGather context parallel forward")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192, help="global sequence length")
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--heads-per-group", type=int, default=2)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--backend", choices=["fa3", "fa4"], default="fa3")
    parser.add_argument("--sm-margins", type=int, nargs="+")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if args.sm_margins is None:
        args.sm_margins = [0, 4, 8, 16] if args.backend == "fa3" else [0]
    assert args.backend == "fa3" or args.sm_margins == [0], "FA4 SM120 仅支持 margin=0"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size >= 2
    assert args.seq_len > 0 and args.seq_len % world_size == 0
    assert args.batch_size > 0 and args.iters > 0 and args.repeats > 0
    assert not (args.check_only and args.skip_check)
    if not args.skip_check:
        check_correctness(group, args.sm_margins, args.backend)
    if args.check_only:
        dist.destroy_process_group()
        return

    torch.manual_seed(200 + rank)
    local_seq = args.seq_len // world_size
    q = torch.randn(args.batch_size, local_seq, args.q_heads, 128, device="cuda").bfloat16()
    k = torch.randn(args.batch_size, local_seq, args.kv_heads, 128, device="cuda").bfloat16()
    v = torch.randn_like(k)
    attention = AllGatherAttention(q, k, args.heads_per_group, args.causal, group, args.backend)
    full_gather = AllGatherAttention(q, k, args.kv_heads, args.causal, group, args.backend)
    if rank == 0:
        print(json.dumps({
            "cp": world_size, "B": args.batch_size, "S": args.seq_len,
            "Hq": args.q_heads, "Hkv": args.kv_heads, "D": 128,
            "heads_per_group": args.heads_per_group, "causal": args.causal,
            "backend": args.backend, "warmup": args.warmup,
            "iters": args.iters, "repeats": args.repeats,
            "torch": torch.__version__, "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(), "sms": attention.num_sms,
            "nccl_env": {name: os.environ.get(name) for name in (
                "NCCL_MIN_CTAS", "NCCL_MAX_CTAS", "NCCL_CGA_CLUSTER_SIZE",
            )},
        }), flush=True)
    cases = [("full_gather", full_gather, False, 0)]
    cases += [
        (mode, attention, mode == "overlap", margin)
        for margin in args.sm_margins
        for mode in ("serial", "overlap")
    ]
    expected = full_gather.forward(q, k, v, False, 0)
    for _, implementation, overlap, margin in cases:
        actual = implementation.forward(q, k, v, overlap, margin)
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
    torch.cuda.synchronize()
    samples = {(mode, margin): [] for mode, _, _, margin in cases}
    # 交替正序和倒序测量，减少温度、clock 和固定测试顺序造成的偏差。
    for repeat in range(args.repeats):
        ordered_cases = cases if repeat % 2 == 0 else list(reversed(cases))
        for mode, implementation, overlap, margin in ordered_cases:
            samples[mode, margin].append(benchmark(
                implementation, q, k, v, overlap, margin, args.warmup, args.iters
            ))
    for mode, implementation, overlap, margin in cases:
        values = samples[mode, margin]
        if rank == 0:
            print(json.dumps({
                "mode": mode, "sm_margin": margin,
                "median_ms": statistics.median(values), "samples_ms": values,
            }), flush=True)
        if args.trace_dir is not None:
            args.trace_dir.mkdir(parents=True, exist_ok=True)
            dist.barrier()
            torch.cuda.synchronize()
            # CPU operator tracing 会放大 launch 间隔，改变短 kernel 的 overlap。
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as profiler:
                # profiler 初始化可能让各 rank 错开；先在 profiler 内再次对齐。
                dist.barrier()
                torch.cuda.synchronize()
                for _ in range(3):
                    implementation.forward(q, k, v, overlap, margin)
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(
                args.trace_dir / f"{mode}-sm{margin}-rank{rank}.json.gz"
            ))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
