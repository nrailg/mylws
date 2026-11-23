import argparse
from pathlib import Path
from statistics import median

import diffusers
import torch
from veomni.models.diffusers.qwen_image.qwen_image_transformer import register_qwen_image_transformer_modeling

from my_qwen_image.transformer_qwenimage import QwenImageTransformer2DModel


@torch.inference_mode()
def benchmark(name: str, model_class: type, args: argparse.Namespace) -> tuple[torch.Tensor, float]:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    model = model_class.from_pretrained(
        args.checkpoint, torch_dtype=dtype, device_map=device, local_files_only=True
    ).eval()
    model.set_attention_backend("native")

    grid_size = args.image_size // 16
    image_tokens = grid_size * grid_size
    generator = torch.Generator(device=device).manual_seed(42)
    inputs = {
        "hidden_states": torch.randn(
            1, image_tokens, model.config.in_channels, device=device, dtype=dtype, generator=generator
        ),
        "encoder_hidden_states": torch.randn(
            1, args.text_tokens, model.config.joint_attention_dim, device=device, dtype=dtype, generator=generator
        ),
        "img_shapes": [[(1, grid_size, grid_size)]],
        "timestep": torch.full((1,), 0.5, device=device, dtype=dtype),
        "return_dict": False,
    }
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"{name} parameters={parameters:,} image_tokens={image_tokens} text_tokens={args.text_tokens}", flush=True)

    for _ in range(args.warmup):
        model(**inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(args.iterations):
        start.record()
        output = model(**inputs)[0]
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))

    latency = median(latencies)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"{name} median_ms={latency:.3f} min_ms={min(latencies):.3f} "
        f"max_ms={max(latencies):.3f} peak_allocated_gib={peak_gib:.3f}",
        flush=True,
    )
    assert output.shape == (1, image_tokens, model.config.patch_size**2 * model.out_channels)
    return output.cpu(), latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Qwen Image transformer inference on one CUDA GPU.")
    parser.add_argument("checkpoint", type=Path, help="Local Diffusers transformer directory with weights and config")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()

    assert (args.checkpoint / "config.json").is_file(), args.checkpoint
    assert args.image_size > 0 and args.image_size % 16 == 0, args.image_size
    assert args.text_tokens > 0 and args.warmup > 0 and args.iterations > 0
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(
        f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} diffusers={diffusers.__version__} "
        f"dtype=bf16 backend=native warmup={args.warmup} iterations={args.iterations} checkpoint={args.checkpoint}",
        flush=True,
    )

    local_output, local_latency = benchmark("my_qwen_image", QwenImageTransformer2DModel, args)
    torch.cuda.empty_cache()
    veomni_model_class = register_qwen_image_transformer_modeling()
    veomni_output, veomni_latency = benchmark("VeOmni", veomni_model_class, args)

    local_output = local_output.float()
    veomni_output = veomni_output.float()
    torch.testing.assert_close(veomni_output, local_output, rtol=0.02, atol=0.02)
    print(
        f"outputs_equal={torch.equal(local_output, veomni_output)} "
        f"max_abs_diff={(local_output - veomni_output).abs().max().item():.6g} "
        f"local_over_veomni={local_latency / veomni_latency:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
