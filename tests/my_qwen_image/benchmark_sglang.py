import argparse
import gc
import json
from collections.abc import Callable
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import diffusers
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from compare_diffusers import report_error
from my_qwen_image.transformer_qwenimage import QwenImageTransformer2DModel


DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
SGLANG_REVISION = "297e54fd105e974cdf401e5b4d20253a22c16f90"
PARITY_TIMESTEPS = (1.0, 0.5, 0.01)
PARITY_RELATIVE_L2 = 0.02
POINTWISE_TOLERANCE = 0.02
Forward = Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]
ImageShapes = list[list[tuple[int, int, int]]]


def load_sglang(checkpoint: Path, revision_path: Path, rendezvous: str) -> torch.nn.Module:
    # 仅在启用 SGLang 模型时加载其依赖。
    import sglang
    from sglang.kernels.ops.diffusion import (
        FUSED_LINEAR_GELU,
        QWEN_IMAGE_ADDED_QKV,
        can_use_linear_gelu,
    )
    from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageDitConfig
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import QwenImagePipelineConfig
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.attention.selector import global_force_attn_backend
    from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
    from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
        QwenImageTransformer2DModel as SGLangTransformer,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import toggle_quality_fusions
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

    # 锁定参与对比的 SGLang revision 和源码位置。
    assert revision_path.read_text().strip() == SGLANG_REVISION
    assert Path(sglang.__file__).resolve().is_relative_to(revision_path.resolve().parent)
    print(f"sglang_revision={SGLANG_REVISION} source={sglang.__file__}", flush=True)

    # 初始化单卡运行环境。
    set_global_server_args(
        ServerArgs(
            model_path=str(checkpoint), pipeline_config=QwenImagePipelineConfig(), num_gpus=1
        )
    )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=1, distributed_init_method=f"file://{rendezvous}/rank0"
    )
    global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)

    # 从同一 checkpoint 加载模型。
    hf_config = json.loads((checkpoint / "config.json").read_text())
    dit_config = QwenImageDitConfig()
    dit_config.update_model_arch(hf_config)
    model = maybe_load_fsdp_model(
        SGLangTransformer,
        {"config": dit_config, "hf_config": hf_config},
        [str(path) for path in sorted(checkpoint.glob("*.safetensors"))],
        DEVICE,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        param_dtype=DTYPE,
        reduce_dtype=torch.float32,
    ).eval()

    # 启用无损融合并确认关键融合已挂载。
    toggle_quality_fusions([model], "lossless")
    qkv_sites = list(QWEN_IMAGE_ADDED_QKV.iter_sites(model))
    gelu_sites = list(FUSED_LINEAR_GELU.iter_sites(model))
    qkv_enabled = sum(map(QWEN_IMAGE_ADDED_QKV.is_enabled, qkv_sites))
    gelu_enabled = sum(map(FUSED_LINEAR_GELU.is_enabled, gelu_sites))
    gelu_input = torch.empty((1, 256, model.inner_dim), device=DEVICE, dtype=DTYPE)
    gelu_eligible = sum(
        FUSED_LINEAR_GELU.is_enabled(site) and can_use_linear_gelu(site.proj, gelu_input)
        for site in gelu_sites
    )
    print(
        f"sglang_quality=lossless added_qkv={qkv_enabled}/{len(qkv_sites)} "
        f"linear_gelu_mounted={gelu_enabled}/{len(gelu_sites)} "
        f"linear_gelu_eligible={gelu_eligible}",
        flush=True,
    )
    assert qkv_enabled == len(model.transformer_blocks)

    return model


def make_forward(
    name: str,
    model: torch.nn.Module,
    img_shapes: ImageShapes,
    text_tokens: int,
    mask: torch.Tensor | None = None,
) -> Callable[..., torch.Tensor]:
    # 适配不同实现的 forward 参数。
    if name == "sglang":
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
            QwenImagePipelineConfig,
        )
        from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

        # 预计算 SGLang 所需的文本长度和 RoPE。
        text_lengths = [text_tokens if mask is None else int(mask.sum())]
        freqs_cis, freqs_complex = QwenImagePipelineConfig.get_freqs_cis(
            img_shapes, [text_tokens], model.rotary_emb, DEVICE, DTYPE
        )

        def forward(
            latents: torch.Tensor, text: torch.Tensor, timestep: torch.Tensor
        ) -> torch.Tensor:
            # 注入 SGLang forward context。
            with set_forward_context(current_timestep=0, attn_metadata=None):
                return model(
                    hidden_states=latents,
                    encoder_hidden_states=text,
                    encoder_hidden_states_mask=mask,
                    timestep=timestep,
                    img_shapes=img_shapes,
                    txt_seq_lens=text_lengths,
                    freqs_cis=freqs_cis,
                    freqs_complex=freqs_complex,
                )

        return forward

    rope_kwargs = {}
    if name == "local":
        rope_kwargs["image_rotary_emb"] = model.pos_embed(
            img_shapes, max_txt_seq_len=text_tokens, device=DEVICE
        )

    def forward(latents: torch.Tensor, text: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # 统一返回裸 tensor。
        return model(
            hidden_states=latents,
            encoder_hidden_states=text,
            encoder_hidden_states_mask=mask,
            timestep=timestep,
            img_shapes=img_shapes,
            return_dict=False,
            **rope_kwargs,
        )[0]

    return forward


def make_workload(
    name: str, model: torch.nn.Module, img_shapes: ImageShapes, text_tokens: int, workload: str
) -> Forward:
    if workload == "single":
        return make_forward(name, model, img_shapes, text_tokens)

    # 本地实现使用融合 CFG，SGLang 分别执行正负条件。
    if name == "local":
        rope = model.pos_embed(img_shapes, max_txt_seq_len=text_tokens, device=DEVICE)

        def pair(
            latents: torch.Tensor,
            text: torch.Tensor,
            negative_text: torch.Tensor,
            timestep: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # 复用融合 CFG 路径。
            return model.forward_cfg(
                latents, text, negative_text, timestep, img_shapes=img_shapes, image_rotary_emb=rope
            )

    else:
        forward = make_forward(name, model, img_shapes, text_tokens)

        def pair(
            latents: torch.Tensor,
            text: torch.Tensor,
            negative_text: torch.Tensor,
            timestep: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # 分别计算正负条件输出。
            return forward(latents, text, timestep), forward(latents, negative_text, timestep)

    return pair


def encode_timestep(name: str, timestep: torch.Tensor) -> torch.Tensor:
    # SGLang 使用放大到 [0, 1000] 的 float32 timestep。
    timestep = timestep.to(DTYPE)
    if name == "sglang":
        return timestep.float() * 1000

    return timestep


def check_outputs(
    name: str, observed: torch.Tensor, expected: torch.Tensor, failures: list[str]
) -> None:
    # 收集误差，避免首个失败掩盖后续结果。
    report_error(name, observed, expected)
    try:
        torch.testing.assert_close(
            observed, expected, rtol=POINTWISE_TOLERANCE, atol=POINTWISE_TOLERANCE
        )
    except AssertionError as error:
        failures.append(f"{name}\n{error}")


@torch.no_grad()
def run_parity(
    name: str,
    model: torch.nn.Module,
    latents: torch.Tensor,
    text: torch.Tensor,
    img_shapes: ImageShapes,
    output_channels: int,
) -> list[torch.Tensor]:
    # 最后一轮额外覆盖 masked text。
    text_tokens = text.shape[1]
    mask = torch.arange(text_tokens, device=DEVICE)[None, :] < text_tokens - 25
    outputs = []
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    with sdpa_kernel(backends, set_priority=True):
        for timestep, text_mask in zip(PARITY_TIMESTEPS, (None, None, mask), strict=True):
            forward = make_forward(name, model, img_shapes, text_tokens, text_mask)
            time_input = encode_timestep(name, torch.tensor([timestep], device=DEVICE))
            output = forward(latents, text, time_input).float().cpu()
            assert output.shape == (*latents.shape[:2], output_channels)
            outputs.append(output)

    return outputs


def check_reference(
    name: str, outputs: list[torch.Tensor], references: list[torch.Tensor], failures: list[str]
) -> list[float]:
    # 同时记录整体误差和逐点偏差。
    errors = []
    for index, (output, reference) in enumerate(zip(outputs, references, strict=True)):
        label = f"{name} timestep={PARITY_TIMESTEPS[index]} masked={index == 2}"
        relative_l2 = report_error(label, output, reference)
        errors.append(relative_l2)
        close = torch.isclose(
            output, reference, rtol=POINTWISE_TOLERANCE, atol=POINTWISE_TOLERANCE
        )
        misses = output.numel() - close.sum().item()
        print(
            f"{label} pointwise_rtol={POINTWISE_TOLERANCE} atol={POINTWISE_TOLERANCE} "
            f"mismatched={misses}/{output.numel()}",
            flush=True,
        )
        if not torch.isfinite(output).all() or not relative_l2 <= PARITY_RELATIVE_L2:
            failures.append(f"{label} relative_l2={relative_l2:.8g}, limit={PARITY_RELATIVE_L2}")

    return errors


def as_tuple(output: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return (output,) if isinstance(output, torch.Tensor) else output


def capture_forward(
    forward: Forward,
    inputs: tuple[torch.Tensor, ...],
    next_inputs: tuple[torch.Tensor, ...],
    name: str,
    failures: list[str],
) -> Forward:
    # CUDA Graph 通过固定地址的输入承载每次的新值。
    static_inputs = tuple(tensor.clone() for tensor in inputs)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            forward(*static_inputs)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = forward(*static_inputs)

    def replay(*values: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # 更新静态输入后重放已捕获的 Graph。
        for target, source in zip(static_inputs, values, strict=True):
            target.copy_(source)
        graph.replay()

        return static_output

    # 用两组输入确认 replay 未固化输入值。
    for probe in (inputs, next_inputs):
        observed = as_tuple(replay(*probe))
        expected = as_tuple(forward(*probe))
        for i, (actual, reference) in enumerate(zip(observed, expected, strict=True)):
            check_outputs(f"{name} graph replay output={i}", actual, reference, failures)

    return replay


def benchmark(
    name: str,
    mode: str,
    workload: str,
    forward: Forward,
    inputs: tuple[torch.Tensor, ...],
    next_inputs: tuple[torch.Tensor, ...],
    timesteps: list[torch.Tensor],
    args: argparse.Namespace,
    failures: list[str],
) -> float:
    # Graph 捕获和预热不计入稳定态耗时。
    label = f"{name} mode={mode} workload={workload}"
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        if mode == "graph":
            forward = capture_forward(forward, inputs, next_inputs, label, failures)

        for timestep in timesteps[:args.warmup]:
            forward(*inputs[:-1], timestep)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        compile_stats = torch._dynamo.utils.counters["stats"]
        graphs_before = compile_stats["unique_graphs"]

        # CUDA Event 只统计 GPU 执行时间。
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        batch_ms = []
        for index in range(args.repeats):
            start.record()
            for timestep in timesteps[index * args.iterations:(index + 1) * args.iterations]:
                forward(*inputs[:-1], timestep)
            end.record()
            end.synchronize()
            batch_ms.append(start.elapsed_time(end) / args.iterations)

        # 计时阶段不应再触发编译。
        if mode == "compile":
            new_graphs = compile_stats["unique_graphs"] - graphs_before
            print(f"{label} compile_scope=transformer_block new_graphs={new_graphs}", flush=True)
            assert new_graphs == 0, f"{label} compiled new graphs during timed batches"

    # 中位数降低偶发抖动的影响。
    median_ms = median(batch_ms)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    forwards = 2 if workload == "cfg" else 1
    print(
        f"{label} median_ms={median_ms:.3f} forwards_per_s={1000 * forwards / median_ms:.3f} "
        f"batch_ms={[round(ms, 3) for ms in batch_ms]} peak_allocated_gib={peak_gib:.3f}",
        flush=True,
    )
    return median_ms


@torch.no_grad()
def measure_model(
    name: str,
    model: torch.nn.Module,
    latents: torch.Tensor,
    text: torch.Tensor,
    negative_text: torch.Tensor,
    img_shapes: ImageShapes,
    args: argparse.Namespace,
    failures: list[str],
) -> dict[tuple[str, str], float]:
    # timestep 数量同时覆盖预热和全部计时批次。
    results = {}
    num_steps = max(args.warmup, args.repeats * args.iterations)
    timesteps = [
        encode_timestep(name, value[None])
        for value in torch.linspace(0.99, 0.01, num_steps, device=DEVICE)
    ]
    probes = {}
    compile_reference = {}

    # 保存 compile 前的 eager 输出作为正确性基准。
    for workload in args.workloads:
        inputs = (latents, text) if workload == "single" else (latents, text, negative_text)
        next_inputs = tuple(tensor.roll(1, dims=1) for tensor in inputs) + (timesteps[-1],)
        probes[workload] = (inputs + (timesteps[0],), next_inputs)
        if "compile" in args.modes:
            forward = make_workload(name, model, img_shapes, args.text_tokens, workload)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                compile_reference[workload] = [
                    tuple(output.cpu() for output in as_tuple(forward(*probe)))
                    for probe in probes[workload]
                ]

    for mode in args.modes:
        # 只编译重复的 Transformer block。
        if mode == "compile":
            for block in model.transformer_blocks:
                block.compile(dynamic=False, options={"triton.cudagraphs": False})

        for workload in args.workloads:
            forward = make_workload(name, model, img_shapes, args.text_tokens, workload)
            inputs, next_inputs = probes[workload]

            # 融合 CFG 应与两次独立 forward 一致。
            if name == "local" and mode == "eager" and workload == "cfg":
                single = make_forward(name, model, img_shapes, args.text_tokens)
                for probe in (inputs, next_inputs, inputs):
                    image, positive_text, negative_text, timestep = probe
                    positive, negative = forward(*probe)
                    expected_positive = single(image, positive_text, timestep)
                    expected_negative = single(image, negative_text, timestep)
                    check_outputs("local CFG positive", positive, expected_positive, failures)
                    check_outputs("local CFG negative", negative, expected_negative, failures)

            if mode == "compile":
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    references = compile_reference[workload]
                    for probe, expected in zip(probes[workload], references, strict=True):
                        observed = as_tuple(forward(*probe))
                        for i, (actual, reference) in enumerate(
                            zip(observed, expected, strict=True)
                        ):
                            label = f"{name} compile workload={workload} output={i}"
                            check_outputs(label, actual.cpu(), reference, failures)

            results[mode, workload] = benchmark(
                name, mode, workload, forward, inputs, next_inputs, timesteps, args, failures
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the Qwen-Image DiT with pinned SGLang on one CUDA GPU."
    )
    parser.add_argument(
        "checkpoint", type=Path, help="Local Diffusers transformer checkpoint directory"
    )
    parser.add_argument(
        "--sglang-revision-file", type=Path, help="REVISION file beside pinned source"
    )
    parser.add_argument(
        "--models", nargs="+", choices=("local", "sglang"), default=("local", "sglang")
    )
    parser.add_argument(
        "--disable-linear-gelu", action="store_true", help="Disable the local linear + GELU fusion"
    )
    modes = ("eager", "graph", "compile")
    workloads = ("single", "cfg")
    parser.add_argument("--modes", nargs="+", choices=modes, default=modes)
    parser.add_argument("--workloads", nargs="+", choices=workloads, default=workloads)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    # Regional compilation 会原地修改模型，因此固定执行顺序。
    args.modes = [mode for mode in modes if mode in args.modes]

    if "sglang" in args.models and args.sglang_revision_file is None:
        parser.error("--sglang-revision-file is required for SGLang")
    assert args.image_size > 0 and args.image_size % 16 == 0
    assert args.text_tokens > 25 and min(args.warmup, args.iterations, args.repeats) > 0
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    config = json.loads((args.checkpoint / "config.json").read_text())
    output_channels = config["patch_size"]**2 * (config["out_channels"] or config["in_channels"])
    grid_size = args.image_size // 16
    img_shapes = [[(1, grid_size, grid_size)]]

    # 固定随机输入，保证各实现使用相同 workload。
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    tensor_kwargs = {"device": DEVICE, "dtype": DTYPE, "generator": generator}
    latents = torch.randn(1, grid_size**2, config["in_channels"], **tensor_kwargs)
    text = torch.randn(1, args.text_tokens, config["joint_attention_dim"], **tensor_kwargs)
    negative_text = torch.randn(text.shape, **tensor_kwargs)

    gpu = torch.cuda.get_device_name(DEVICE)
    capability = torch.cuda.get_device_capability(DEVICE)
    print(
        f"gpu={gpu} capability={capability} "
        f"torch={torch.__version__} cuda={torch.version.cuda} diffusers={diffusers.__version__} "
        f"dtype=bf16 timing_backend=sdpa_flash batch=1 image_tokens={latents.shape[1]} "
        f"text_tokens={text.shape[1]} warmup={args.warmup} iterations={args.iterations} "
        f"repeats={args.repeats} checkpoint={args.checkpoint}",
        flush=True,
    )

    # Diffusers 提供统一的数值基准。
    failures = []
    reference = diffusers.QwenImageTransformer2DModel.from_pretrained(
        args.checkpoint, torch_dtype=DTYPE, device_map=DEVICE, local_files_only=True
    ).eval()
    reference.set_attention_backend("native")
    expected_parameters = sum(parameter.numel() for parameter in reference.parameters())
    print(f"diffusers parameters={expected_parameters:,}", flush=True)
    reference_outputs = run_parity(
        "diffusers", reference, latents, text, img_shapes, output_channels
    )
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    # 逐一检查实现的数值一致性和性能。
    results = {}
    parity_errors = {}
    rendezvous = TemporaryDirectory()
    for name in args.models:
        if name == "sglang":
            model = load_sglang(args.checkpoint, args.sglang_revision_file, rendezvous.name)
        else:
            model = QwenImageTransformer2DModel.from_pretrained(
                args.checkpoint, torch_dtype=DTYPE, device_map=DEVICE, local_files_only=True
            ).eval()
            model.set_attention_backend("native")
            model.enable_inference_optimizations(fuse_linear_gelu=not args.disable_linear_gelu)
            enabled_blocks = sum(block._fuse_linear_gelu for block in model.transformer_blocks)
            print(
                f"local_linear_gelu_enabled="
                f"{2 * enabled_blocks}/{2 * len(model.transformer_blocks)} sites",
                flush=True,
            )

        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"{name} parameters={parameters:,}", flush=True)
        assert parameters == expected_parameters
        outputs = run_parity(name, model, latents, text, img_shapes, output_channels)
        parity_errors[name] = check_reference(name, outputs, reference_outputs, failures)
        results[name] = measure_model(
            name, model, latents, text, negative_text, img_shapes, args, failures
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 延迟之比等价于 local 相对 SGLang 的吞吐率。
    for mode in args.modes:
        for workload in args.workloads:
            if "local" in results and "sglang" in results:
                ratio = results["sglang"][mode, workload] / results["local"][mode, workload]
                print(
                    f"mode={mode} workload={workload} local_over_sglang_throughput={ratio:.4f}",
                    flush=True,
                )

    # 释放 SGLang 创建的分布式状态。
    if torch.distributed.is_initialized():
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    rendezvous.cleanup()

    # local 的误差不得高于参与对比的 SGLang。
    if "local" in parity_errors and "sglang" in parity_errors:
        for timestep, local_l2, sglang_l2 in zip(
            PARITY_TIMESTEPS, parity_errors["local"], parity_errors["sglang"], strict=True
        ):
            if local_l2 > sglang_l2:
                failures.append(
                    f"timestep={timestep} local_relative_l2={local_l2:.8g} "
                    f"exceeds sglang_relative_l2={sglang_l2:.8g}"
                )

    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    main()
