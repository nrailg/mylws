import argparse
from collections.abc import Callable
from pathlib import Path
from statistics import median

import diffusers
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from my_qwen_image.transformer_qwenimage import QwenImageTransformer2DModel


DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
RELATIVE_L2_LIMIT = 0.02
CASES = ("baseline", "projection", "text", "batch-reference", "batch-fused", "batch-cfg")
Schedule = Callable[..., torch.Tensor]


def make_schedule(
    model: QwenImageTransformer2DModel,
    rotary: tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
    workload: str,
    case: str,
) -> Schedule:
    reuse_projection = case in ("projection", "text")
    reuse_first_block = case == "text"

    def schedule(
        latents: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if reuse_projection:
            positive_text = model.prepare_inference_text(positive_text)
            if workload == "cfg":
                negative_text = model.prepare_inference_text(negative_text)
        elif batch_size > 1:
            positive_text = positive_text.expand(batch_size, -1, -1)
            negative_text = negative_text.expand(batch_size, -1, -1)
        if case == "batch-cfg":
            cfg_text = torch.cat((positive_text, negative_text))

        predictions = []
        for step_latents, timestep in zip(latents.unbind(), timesteps.unbind(), strict=True):
            positive, negative = positive_text, negative_text
            if reuse_first_block:
                positive = model.prepare_inference_first_block_text(positive, timestep, rotary)
                if workload == "cfg":
                    negative = model.prepare_inference_first_block_text(negative, timestep, rotary)
            if batch_size > 1 and case != "batch-cfg":
                timestep = timestep.expand(batch_size)

            branches = []
            for images in step_latents.split(batch_size):
                if case == "batch-cfg":
                    (prediction,) = model(
                        torch.cat((images, images)),
                        cfg_text,
                        timestep=timestep.expand(2 * batch_size),
                        image_rotary_emb=rotary,
                        return_dict=False,
                    )
                    outputs = prediction.split(batch_size)
                elif workload == "cfg":
                    outputs = model.forward_cfg(
                        images, positive, negative, timestep, image_rotary_emb=rotary
                    )
                else:
                    outputs = model(
                        images, positive, timestep=timestep, image_rotary_emb=rotary, return_dict=False
                    )
                branches.append(outputs)
            step_outputs = [torch.cat(branch) for branch in zip(*branches, strict=True)]
            predictions.append(torch.stack(step_outputs))
        return torch.stack(predictions)

    return schedule


def report_parity(label: str, output: torch.Tensor, reference: torch.Tensor) -> bool:
    assert output.shape == reference.shape, (output.shape, reference.shape)
    # Reduce per step, branch, and image before reporting the worst result.
    actual = output.float().flatten(-2)
    expected = reference.float().flatten(-2)
    error = actual - expected
    max_abs = error.abs().max().item()
    rmse = error.square().mean(dim=-1).sqrt().max().item()
    relative_l2 = (
        torch.linalg.vector_norm(error, dim=-1) / torch.linalg.vector_norm(expected, dim=-1)
    ).max().item()
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1).min().item()
    finite = torch.isfinite(actual).all().item()
    bitwise = torch.equal(output, reference)
    print(
        f"{label} sample_count={actual.numel() // actual.shape[-1]} finite={finite} "
        f"bitwise={bitwise} max_abs={max_abs:.8g} max_rmse={rmse:.8g} "
        f"max_relative_l2={relative_l2:.8g} min_cosine={cosine:.8g}",
        flush=True,
    )
    return finite and relative_l2 <= RELATIVE_L2_LIMIT


def timed_schedule(
    schedule: Schedule,
    inputs: tuple[torch.Tensor, ...],
    mode: str,
    label: str,
    warmup: int,
    repeats: int,
) -> tuple[torch.Tensor, float]:
    torch.cuda.empty_cache()
    if mode == "graph":
        static_inputs = tuple(tensor.clone() for tensor in inputs)
        latents, positive, negative, timesteps = inputs
        probe_inputs = (
            latents.roll(1, dims=1), positive.roll(1, dims=1), negative.roll(1, dims=1),
            timesteps * 0.75,
        )
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup):
                expected = schedule(*probe_inputs).cpu()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = schedule(*static_inputs)

        def run(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
            for target, source in zip(static_inputs, values, strict=True):
                target.copy_(source)
            graph.replay()
            return static_output

        observed = run(probe_inputs).cpu()
        assert torch.equal(observed, expected), f"{label} changed-input graph replay differs from eager"
        print(f"{label} changed_input_graph_replay=bitwise", flush=True)
        del probe_inputs
    else:
        def run(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
            return schedule(*values)

        for _ in range(warmup):
            run(inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(repeats):
        start.record()
        output = run(inputs)
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))

    schedule_ms = median(latencies)
    steps, branches, group_size = output.shape[:3]
    image_steps_per_second = 1000 * steps * group_size / schedule_ms
    cfg_rate = f" cfg_pairs_per_s={image_steps_per_second:.4f}" if branches == 2 else ""
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    print(
        f"{label} schedule_ms={schedule_ms:.3f} group_step_ms={schedule_ms / steps:.3f} "
        f"dit_samples_per_s={branches * image_steps_per_second:.4f}{cfg_rate} "
        f"samples_ms={[round(value, 3) for value in latencies]} "
        f"peak_allocated_gib={peak_allocated:.3f} peak_reserved_gib={peak_reserved:.3f}",
        flush=True,
    )
    return output.cpu(), schedule_ms


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure same-prompt Qwen Image DiT group inference.")
    parser.add_argument("checkpoint", type=Path)
    default_cases = tuple(case for case in CASES if case not in ("projection", "batch-cfg"))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=default_cases)
    parser.add_argument("--modes", nargs="+", choices=("eager", "graph"), default=("eager",))
    workloads = ("single", "cfg")
    parser.add_argument("--workloads", nargs="+", choices=workloads, default=workloads)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timesteps", nargs="+", type=float, default=(0.99, 0.5, 0.01))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    assert (args.checkpoint / "config.json").is_file(), args.checkpoint
    assert args.image_size > 0 and args.image_size % 16 == 0
    assert min(args.text_tokens, args.warmup, args.repeats) > 0
    assert 1 < args.batch_size <= args.group_size and args.group_size % args.batch_size == 0
    assert torch.cuda.is_bf16_supported()
    print(
        f"gpu={torch.cuda.get_device_name(DEVICE)} torch={torch.__version__} "
        f"cuda={torch.version.cuda} diffusers={diffusers.__version__} dtype=bf16 "
        f"sdpa_backend=flash group_size={args.group_size} batch_size={args.batch_size} "
        f"timesteps={list(args.timesteps)} warmup={args.warmup} repeats={args.repeats} seed=42 "
        f"max_relative_l2_limit={RELATIVE_L2_LIMIT} checkpoint={args.checkpoint}",
        flush=True,
    )
    model = QwenImageTransformer2DModel.from_pretrained(
        args.checkpoint, torch_dtype=DTYPE, device_map=DEVICE, local_files_only=True
    ).eval()
    model.set_attention_backend("native")
    model.enable_inference_optimizations(fuse_linear_gelu=True)
    processors = tuple(block.attn.processor for block in model.transformer_blocks)
    processor_kernels = tuple(processor._inference_kernels for processor in processors)

    grid_size = args.image_size // 16
    image_tokens = grid_size**2
    rotary = model.prepare_inference_rope([[(1, grid_size, grid_size)]], args.text_tokens)
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    tensor_kwargs = {"device": DEVICE, "dtype": DTYPE, "generator": generator}
    latents = torch.randn(
        len(args.timesteps), args.group_size, image_tokens, model.config.in_channels, **tensor_kwargs
    )
    text_shape = (1, args.text_tokens, model.config.joint_attention_dim)
    positive_text = torch.randn(text_shape, **tensor_kwargs)
    negative_text = torch.randn(text_shape, **tensor_kwargs)
    timesteps = torch.tensor(args.timesteps, device=DEVICE, dtype=DTYPE)[:, None]
    inputs = (latents, positive_text, negative_text, timesteps)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    gelu_blocks = sum(block._fuse_linear_gelu for block in model.transformer_blocks)
    print(
        f"parameters={parameters} image_tokens={image_tokens} text_tokens={args.text_tokens} "
        f"linear_gelu_sites={2 * gelu_blocks}/{2 * len(processors)}",
        flush=True,
    )

    failures = []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for workload in args.workloads:
            reference = None
            if "eager" not in args.modes or "baseline" not in args.cases:
                baseline = make_schedule(model, rotary, 1, workload, "baseline")
                reference = baseline(*inputs).cpu()
            for mode in ("eager", "graph"):
                if mode not in args.modes:
                    continue
                latencies = {}
                batch_reference = None
                for case in CASES:
                    if case not in args.cases or (case == "batch-cfg" and workload != "cfg"):
                        continue
                    # Disable only the joint attention fusion in the same-batch control.
                    for processor, kernels in zip(processors, processor_kernels, strict=True):
                        processor._inference_kernels = None if case == "batch-reference" else kernels
                    batched = case in ("batch-reference", "batch-fused", "batch-cfg")
                    batch_size = args.batch_size if batched else 1
                    schedule = make_schedule(model, rotary, batch_size, workload, case)
                    label = f"case={case} mode={mode} workload={workload} batch={batch_size}"
                    if case == "batch-cfg":
                        label += f" effective_dit_batch={2 * batch_size}"
                    print(f"running {label}", flush=True)
                    output, latency = timed_schedule(
                        schedule, inputs, mode, label, args.warmup, args.repeats
                    )
                    latencies[case] = latency
                    if case == "baseline" and mode == "eager":
                        reference = output
                    if case == "batch-reference":
                        batch_reference = output
                    if "baseline" in latencies:
                        speedup = latencies["baseline"] / latency
                        print(f"{label} throughput_vs_baseline={speedup:.5f}", flush=True)
                    if case == "batch-fused" and "batch-reference" in latencies:
                        speedup = latencies["batch-reference"] / latency
                        print(f"{label} throughput_vs_batch_reference={speedup:.5f}", flush=True)
                        assert batch_reference is not None
                        report_parity(f"{label} parity_reference=batch-reference", output, batch_reference)
                    assert reference is not None
                    if not report_parity(label, output, reference):
                        failures.append(label)
                for processor, kernels in zip(processors, processor_kernels, strict=True):
                    processor._inference_kernels = kernels
    assert not failures, f"Numerical parity failed: {failures}"


if __name__ == "__main__":
    main()
