import json
from collections.abc import Callable
from pathlib import Path

import sglang
import torch
from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageDitConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import QwenImagePipelineConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import global_force_attn_backend
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel as SGLangTransformer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import toggle_quality_fusions
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
SGLANG_REVISION = "297e54fd105e974cdf401e5b4d20253a22c16f90"


def load_sglang(checkpoint: Path, rendezvous: str) -> torch.nn.Module:
    revision_path = Path(sglang.__file__).resolve().parents[2] / "REVISION"
    assert revision_path.read_text().strip() == SGLANG_REVISION
    print(f"sglang_revision={SGLANG_REVISION} source={sglang.__file__}", flush=True)

    set_global_server_args(
        ServerArgs(model_path=str(checkpoint), pipeline_config=QwenImagePipelineConfig(), num_gpus=1)
    )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=1, distributed_init_method=f"file://{rendezvous}/rank0"
    )
    global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)
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

    toggle_quality_fusions([model], "lossless")
    return model


def make_sglang_forward(
    model: torch.nn.Module,
    img_shapes: list[list[tuple[int, int, int]]],
    text_tokens: int,
    mask: torch.Tensor | None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    text_lengths = [text_tokens if mask is None else int(mask.sum())]
    freqs_cis, freqs_complex = QwenImagePipelineConfig.get_freqs_cis(
        img_shapes, [text_tokens], model.rotary_emb, DEVICE, DTYPE
    )

    def forward(latents: torch.Tensor, text: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
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
