import argparse
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from diffusers import QwenImagePipeline
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration


GROUP_SIZE = 8
WARMUP_GROUPS = 2
MEASURED_GROUPS = 3
DEVICE = torch.device("cuda:0")
PROMPTS = (
    "A red fox in a snowy pine forest under soft morning light, detailed wildlife photography.",
    "blurry, low resolution, distorted anatomy",
)

PromptEncoding = tuple[torch.Tensor, torch.Tensor | None]
EncodedGroup = tuple[PromptEncoding, ...]


def encode_independently(pipe: QwenImagePipeline) -> EncodedGroup:
    group = []
    for prompt in PROMPTS:
        encodings = [pipe.encode_prompt(prompt, device=DEVICE) for _ in range(GROUP_SIZE)]
        embeddings, masks = zip(*encodings)
        group_mask = None if masks[0] is None else torch.cat(masks)
        group.append((torch.cat(embeddings), group_mask))
    return tuple(group)


def encode_once(pipe: QwenImagePipeline) -> EncodedGroup:
    return tuple(
        pipe.encode_prompt(prompt, device=DEVICE, num_images_per_prompt=GROUP_SIZE)
        for prompt in PROMPTS
    )


def verify_encodings(baseline_group: EncodedGroup, reused_group: EncodedGroup) -> None:
    for name, baseline, reused in zip(("positive", "negative"), baseline_group, reused_group):
        baseline_embeddings, baseline_mask = baseline
        reused_embeddings, reused_mask = reused
        assert baseline_embeddings.shape[0] == GROUP_SIZE
        assert baseline_embeddings.dtype == torch.bfloat16
        torch.testing.assert_close(reused_embeddings, baseline_embeddings, atol=0, rtol=0)
        if baseline_mask is None:
            assert reused_mask is None
            valid_tokens = baseline_embeddings.size(1)
            mask_shape = None
        else:
            torch.testing.assert_close(reused_mask, baseline_mask, atol=0, rtol=0)
            valid_tokens = baseline_mask[0].sum().item()
            mask_shape = tuple(baseline_mask.shape)
        print(
            f"{name} valid_tokens={valid_tokens} "
            f"embeddings={tuple(baseline_embeddings.shape)} mask={mask_shape}"
        )


def measure(
    encode_group: Callable[[QwenImagePipeline], EncodedGroup], pipe: QwenImagePipeline
) -> tuple[float, int]:
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    start = perf_counter()
    group = encode_group(pipe)
    torch.cuda.synchronize(DEVICE)
    elapsed_ms = (perf_counter() - start) * 1000
    peak_bytes = torch.cuda.max_memory_allocated(DEVICE)
    del group
    return elapsed_ms, peak_bytes


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Qwen-Image same-prompt encoder reuse.")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    assert torch.cuda.device_count() == 1
    print(f"device={torch.cuda.get_device_name(DEVICE)} torch={torch.__version__}", flush=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(args.checkpoint / "tokenizer", local_files_only=True)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.checkpoint / "text_encoder",
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": str(DEVICE)},
    ).eval()
    pipe = QwenImagePipeline(
        scheduler=None, vae=None, text_encoder=text_encoder, tokenizer=tokenizer, transformer=None
    )

    for _ in range(WARMUP_GROUPS - 1):
        encode_independently(pipe)
        encode_once(pipe)
    verify_encodings(encode_independently(pipe), encode_once(pipe))

    baseline_runs = []
    reused_runs = []
    model_bytes = torch.cuda.memory_allocated(DEVICE)
    for _ in range(MEASURED_GROUPS):
        baseline_runs.append(measure(encode_independently, pipe))
        reused_runs.append(measure(encode_once, pipe))

    print(f"group={GROUP_SIZE} warmups={WARMUP_GROUPS} samples={MEASURED_GROUPS} equality=exact")
    print(f"model_allocated_gib={model_bytes / 2**30:.3f}")
    for name, runs in (("independent", baseline_runs), ("reused", reused_runs)):
        timings, peaks = zip(*runs)
        samples = ", ".join(f"{elapsed:.3f}" for elapsed in timings)
        print(
            f"{name} encoder_and_embedding_prep_group_ms=[{samples}] "
            f"median_ms={median(timings):.3f} peak_allocated_gib={max(peaks) / 2**30:.3f}"
        )


if __name__ == "__main__":
    main()
