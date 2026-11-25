import argparse
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import diffusers
import torch
from diffusers import QwenImageTransformer2DModel as DiffusersTransformer

from my_qwen_image.transformer_qwenimage import QwenImageTransformer2DModel


DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
IMG_SHAPES = [[(1, 64, 64)]]
SGLANG_COSINE_THRESHOLD = 0.995
Case = tuple[float, torch.Tensor | None]


def report_error(name: str, observed: torch.Tensor, expected: torch.Tensor) -> float:
    error = observed.float() - expected.float()
    reference_norm = torch.linalg.vector_norm(expected.float())
    relative_l2 = (torch.linalg.vector_norm(error) / reference_norm).item()
    print(
        f"{name} equal={torch.equal(observed, expected)} "
        f"max_abs={error.abs().max().item():.8g} rmse={error.square().mean().sqrt().item():.8g} "
        f"relative_l2={relative_l2:.8g}",
        flush=True,
    )
    return relative_l2


def checked_output(output: torch.Tensor, expected_shape: tuple[int, ...]) -> torch.Tensor:
    assert output.shape == expected_shape, (output.shape, expected_shape)
    output = output.float().cpu()
    assert torch.isfinite(output).all(), "DiT output contains non-finite values"
    return output


@torch.inference_mode()
def run_diffusers_model(
    model_class: type,
    checkpoint: Path,
    latents: torch.Tensor,
    text: torch.Tensor,
    cases: list[Case],
    output_shape: tuple[int, ...],
) -> list[torch.Tensor]:
    # Run identical DiT inputs with and without a text padding mask.
    model = model_class.from_pretrained(
        checkpoint, torch_dtype=DTYPE, device_map=DEVICE, local_files_only=True
    ).eval()
    model.set_attention_backend("native")
    outputs = []
    for timestep, text_mask in cases:
        output = model(
            hidden_states=latents,
            encoder_hidden_states=text,
            encoder_hidden_states_mask=text_mask,
            img_shapes=IMG_SHAPES,
            timestep=torch.tensor([timestep], device=DEVICE, dtype=DTYPE),
            return_dict=False,
        )[0]
        outputs.append(checked_output(output, output_shape))
        print(f"{model_class.__module__} timestep={timestep} completed", flush=True)
    del model, output
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


@torch.no_grad()
def compare_sglang(
    checkpoint: Path,
    rendezvous: str,
    latents: torch.Tensor,
    text: torch.Tensor,
    cases: list[Case],
    output_shape: tuple[int, ...],
    reference: list[torch.Tensor],
    actual: list[torch.Tensor],
) -> None:
    # Keep SGLang optional for Diffusers-only runs.
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import toggle_quality_fusions

    from sglang_reference import load_sglang, make_sglang_forward

    model = load_sglang(checkpoint, rendezvous)

    def compare_case(index: int, quality: str) -> float:
        timestep, mask = cases[index]
        forward = make_sglang_forward(model, IMG_SHAPES, text.shape[1], mask)
        # SGLang expects scheduler units and divides by 1000 before the BF16 cast.
        time_input = torch.tensor([timestep], device=DEVICE, dtype=DTYPE).float() * 1000
        expected = checked_output(forward(latents, text, time_input), output_shape)
        observed = actual[index]
        label = f"case={index} sglang_quality={quality}"
        cosine = torch.nn.functional.cosine_similarity(observed.flatten(), expected.flatten(), dim=0).item()
        reference_cosine = torch.nn.functional.cosine_similarity(
            reference[index].flatten(), expected.flatten(), dim=0
        ).item()
        report_error(f"{label} local_vs_sglang cosine={cosine:.8g}", observed, expected)
        report_error(f"{label} diffusers_vs_sglang cosine={reference_cosine:.8g}", reference[index], expected)
        return cosine

    failed_cases = []
    for index in range(len(cases)):
        if compare_case(index, "lossless") <= SGLANG_COSINE_THRESHOLD:
            failed_cases.append(index)

    if failed_cases:
        toggle_quality_fusions([model], "exact")
        for index in failed_cases:
            compare_case(index, "exact")
        raise AssertionError(f"SGLang lossless cosine must exceed {SGLANG_COSINE_THRESHOLD}: {failed_cases}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local Qwen Image DiT with Diffusers and SGLang.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sglang", action="store_true", help="Also compare with the pinned SGLang DiT.")
    args = parser.parse_args()
    print(
        f"gpu={torch.cuda.get_device_name(DEVICE)} torch={torch.__version__} "
        f"diffusers={diffusers.__version__} dtype=bf16 backend=native seed=42",
        flush=True,
    )
    config = json.loads((args.checkpoint / "config.json").read_text())
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    latents = torch.randn(1, 4096, config["in_channels"], device=DEVICE, dtype=DTYPE, generator=generator)
    text = torch.randn(1, 256, config["joint_attention_dim"], device=DEVICE, dtype=DTYPE, generator=generator)
    mask = torch.arange(256, device=DEVICE)[None, :] < 231
    cases = [(1.0, None), (0.5, None), (0.01, mask)]
    output_channels = config["out_channels"] or config["in_channels"]
    output_shape = (1, 4096, config["patch_size"] ** 2 * output_channels)
    reference = run_diffusers_model(DiffusersTransformer, args.checkpoint, latents, text, cases, output_shape)
    actual = run_diffusers_model(
        QwenImageTransformer2DModel, args.checkpoint, latents, text, cases, output_shape
    )
    for index, (expected, observed) in enumerate(zip(reference, actual, strict=True)):
        report_error(f"case={index} local_vs_diffusers", observed, expected)

    if args.sglang:
        from sglang.multimodal_gen.runtime.distributed.parallel_state import cleanup_dist_env_and_memory

        # Keep the file store available until the process groups are destroyed.
        with TemporaryDirectory() as rendezvous:
            try:
                compare_sglang(
                    args.checkpoint, rendezvous, latents, text, cases, output_shape, reference, actual
                )
            finally:
                if torch.distributed.is_initialized():
                    cleanup_dist_env_and_memory()

    for expected, observed in zip(reference, actual, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


if __name__ == "__main__":
    main()
