# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from math import prod
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.activations import GELU
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm
from diffusers.utils import apply_lora_scale, logging
from diffusers.utils.torch_utils import lru_cache_unless_export, maybe_allow_in_graph


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def compute_text_seq_len_from_mask(
    encoder_hidden_states: torch.Tensor, encoder_hidden_states_mask: torch.Tensor | None
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Compute text sequence length without assuming contiguous masks. Returns length for RoPE and a normalized bool mask.
    """
    batch_size, text_seq_len = encoder_hidden_states.shape[:2]
    if encoder_hidden_states_mask is None:
        return text_seq_len, None, None

    if encoder_hidden_states_mask.shape[:2] != (batch_size, text_seq_len):
        raise ValueError(
            f"`encoder_hidden_states_mask` shape {encoder_hidden_states_mask.shape} must match "
            f"(batch_size, text_seq_len)=({batch_size}, {text_seq_len})."
        )

    if encoder_hidden_states_mask.dtype != torch.bool:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)

    position_ids = torch.arange(text_seq_len, device=encoder_hidden_states.device, dtype=torch.long)
    active_positions = torch.where(encoder_hidden_states_mask, position_ids, position_ids.new_zeros(()))
    has_active = encoder_hidden_states_mask.any(dim=1)
    per_sample_len = torch.where(
        has_active,
        active_positions.max(dim=1).values + 1,
        torch.as_tensor(text_seq_len, device=encoder_hidden_states.device),
    )
    return text_seq_len, per_sample_len, encoder_hidden_states_mask


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, use_additional_t_cond=False):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond=None) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return conditioning


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    @lru_cache_unless_export(maxsize=None)
    def _get_device_freqs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pos_freqs and neg_freqs on the given device."""
        return self.pos_freqs.to(device), self.neg_freqs.to(device)

    def forward(
        self,
        video_fhw: tuple[int, int, int, list[tuple[int, int, int]]],
        device: torch.device = None,
        max_txt_seq_len: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_fhw (`tuple[int, int, int]` or `list[tuple[int, int, int]]`):
                A list of 3 integers [frame, height, width] representing the shape of the video.
            device: (`torch.device`, *optional*):
                The device on which to perform the RoPE computation.
            max_txt_seq_len (`int` or `torch.Tensor`, *optional*):
                The maximum text sequence length for RoPE computation. This should match the encoder hidden states
                sequence length. Can be either an int or a scalar tensor (for torch.compile compatibility).
        """
        if max_txt_seq_len is None:
            raise ValueError("`max_txt_seq_len` must be provided.")

        # Validate batch inference with variable-sized images
        if isinstance(video_fhw, list) and len(video_fhw) > 1:
            # Check if all instances have the same size
            first_fhw = video_fhw[0]
            if not all(fhw == first_fhw for fhw in video_fhw):
                logger.warning(
                    "Batch inference with variable-sized images is not currently supported in QwenEmbedRope. "
                    "All images in the batch should have the same dimensions (frame, height, width). "
                    f"Detected sizes: {video_fhw}. Using the first image's dimensions {first_fhw} "
                    "for RoPE computation, which may lead to incorrect results for other images in the batch."
                )

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            # RoPE frequencies are cached via a lru_cache decorator on _compute_video_freqs
            video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_txt_seq_len_int = int(max_txt_seq_len)
        # Use cached device-transferred freqs to avoid CPU→GPU sync every forward call
        pos_freqs_device, _ = self._get_device_freqs(device)
        txt_freqs = pos_freqs_device[max_vid_index : max_vid_index + max_txt_seq_len_int, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @lru_cache_unless_export(maxsize=128)
    def _compute_video_freqs(
        self, frame: int, height: int, width: int, idx: int = 0, device: torch.device = None
    ) -> torch.Tensor:
        seq_lens = frame * height * width
        pos_freqs, neg_freqs = (
            self._get_device_freqs(device) if device is not None else (self.pos_freqs, self.neg_freqs)
        )

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenEmbedLayer3DRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    @lru_cache_unless_export(maxsize=None)
    def _get_device_freqs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pos_freqs and neg_freqs on the given device."""
        return self.pos_freqs.to(device), self.neg_freqs.to(device)

    def forward(
        self,
        video_fhw: tuple[int, int, int, list[tuple[int, int, int]]],
        max_txt_seq_len: int | torch.Tensor,
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_fhw (`tuple[int, int, int]` or `list[tuple[int, int, int]]`):
                A list of 3 integers [frame, height, width] representing the shape of the video, or a list of layer
                structures.
            max_txt_seq_len (`int` or `torch.Tensor`):
                The maximum text sequence length for RoPE computation. This should match the encoder hidden states
                sequence length. Can be either an int or a scalar tensor (for torch.compile compatibility).
            device: (`torch.device`, *optional*):
                The device on which to perform the RoPE computation.
        """
        # Validate batch inference with variable-sized images
        # In Layer3DRope, the outer list represents batch, inner list/tuple represents layers
        if isinstance(video_fhw, list) and len(video_fhw) > 1:
            # Check if this is batch inference (list of layer lists/tuples)
            first_entry = video_fhw[0]
            if not all(entry == first_entry for entry in video_fhw):
                logger.warning(
                    "Batch inference with variable-sized images is not currently supported in QwenEmbedLayer3DRope. "
                    "All images in the batch should have the same layer structure. "
                    f"Detected sizes: {video_fhw}. Using the first image's layer structure {first_entry} "
                    "for RoPE computation, which may lead to incorrect results for other images in the batch."
                )

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx != layer_num:
                video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            else:
                ### For the condition image, we set the layer index to -1
                video_freq = self._compute_condition_freqs(frame, height, width, device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        max_txt_seq_len_int = int(max_txt_seq_len)
        # Use cached device-transferred freqs to avoid CPU→GPU sync every forward call
        pos_freqs_device, _ = self._get_device_freqs(device)
        txt_freqs = pos_freqs_device[max_vid_index : max_vid_index + max_txt_seq_len_int, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @lru_cache_unless_export(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0, device: torch.device = None):
        seq_lens = frame * height * width
        pos_freqs, neg_freqs = (
            self._get_device_freqs(device) if device is not None else (self.pos_freqs, self.neg_freqs)
        )

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()

    @lru_cache_unless_export(maxsize=None)
    def _compute_condition_freqs(self, frame, height, width, device: torch.device = None):
        seq_lens = frame * height * width
        pos_freqs, neg_freqs = (
            self._get_device_freqs(device) if device is not None else (self.pos_freqs, self.neg_freqs)
        )

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


def _use_inference_kernels(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled("cuda")
    )


def _use_plain_linear(layer: nn.Module, x: torch.Tensor) -> bool:
    return (
        type(layer) is nn.Linear
        and type(layer.weight) is nn.Parameter
        and layer.weight.dtype == x.dtype
        and layer.weight.device == x.device
        and layer.bias is not None
        and layer.bias.dtype == x.dtype
        and layer.bias.device == x.device
        and not layer._forward_pre_hooks
        and not layer._forward_hooks
    )


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None
    _parallel_config = None
    _inference_kernels: ModuleType | None = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def _joint_qkv(
        self,
        attn: Attention,
        image: torch.Tensor,
        text: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
        masked: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        kernels = self._inference_kernels
        if (
            kernels is None
            or attn.training
            or self._parallel_config is not None
            or not _use_inference_kernels(image)
            or image.shape[0] == 0
            or image.shape[1] == 0
            or text.shape[1] == 0
            or not text.is_contiguous()
            or text.dtype != image.dtype
            or rope is None
        ):
            return None

        image_projections = (attn.to_q, attn.to_k, attn.to_v)
        text_projections = (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)
        if not all(_use_plain_linear(layer, image) for layer in (*image_projections, *text_projections)):
            return None

        norms = (attn.norm_q, attn.norm_k, attn.norm_added_q, attn.norm_added_k)
        head_dim = attn.inner_dim // attn.heads
        if not all(
            type(norm) is RMSNorm
            and norm.bias is None
            and norm.weight is not None
            and norm.weight.shape == (head_dim,)
            and norm.weight.dtype == image.dtype
            and norm.weight.device == image.device
            and norm.weight.is_contiguous()
            for norm in norms
        ):
            return None
        if any(
            freq.dtype != torch.complex64
            or freq.device != image.device
            or freq.shape != (stream.shape[1], head_dim // 2)
            or freq.stride(-1) != 1
            for freq, stream in zip(rope, (image, text))
        ):
            return None

        image_weights = [layer.weight for layer in image_projections]
        image_biases = [layer.bias for layer in image_projections]
        text_weights = [layer.weight for layer in text_projections]
        text_biases = [layer.bias for layer in text_projections]
        batch_size = image.shape[0]
        if batch_size == 1:
            image_rows, text_rows = image[0], text[0]
        else:
            # Sequence-major rows keep each stream contiguous across the batch.
            image_rows = image.transpose(0, 1).flatten(0, 1)
            text_rows = text.transpose(0, 1).flatten(0, 1)
        query, key, value = kernels.joint_qkv(
            image_rows, text_rows, image_weights, image_biases, text_weights, text_biases,
            masked, rope[0], rope[1],
            [norm.weight for norm in norms], [norm.eps for norm in norms],
            batch_size,
        )
        if batch_size == 1:
            return tuple(buf.unflatten(-1, (-1, head_dim)).unsqueeze(0) for buf in (query, key, value))
        return tuple(
            buf.view(-1, batch_size, buf.shape[1] // head_dim, head_dim).transpose(0, 1)
            for buf in (query, key, value)
        )

    @staticmethod
    def _reference_joint_qkv(
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape by a fixed `head_dim` and let `-1` absorb the head count. Under tensor parallelism each rank
        # holds a column-sharded slice (`attn.heads // tp_degree` heads); this keeps the processor TP-agnostic.
        # The shared Attention has no `head_dim` attribute; derive it from the unsharded `inner_dim`/`heads`.
        head_dim = attn.inner_dim // attn.heads
        img_query = img_query.unflatten(-1, (-1, head_dim))
        img_key = img_key.unflatten(-1, (-1, head_dim))
        img_value = img_value.unflatten(-1, (-1, head_dim))

        txt_query = txt_query.unflatten(-1, (-1, head_dim))
        txt_key = txt_key.unflatten(-1, (-1, head_dim))
        txt_value = txt_value.unflatten(-1, (-1, head_dim))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        return joint_query, joint_key, joint_value

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        if attention_mask is not None:
            raise ValueError(
                "QwenDoubleStreamAttnProcessor2_0 does not accept an external attention_mask. "
                "Pass encoder_hidden_states_mask to let the processor build the joint mask."
            )

        if encoder_hidden_states_mask is not None:
            seq_img = hidden_states.shape[1]
            image_mask = torch.ones((hidden_states.shape[0], seq_img), dtype=torch.bool, device=hidden_states.device)
            attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            attention_mask = attention_mask[:, None, None, :]

        seq_txt = encoder_hidden_states.shape[1]

        joint_qkv = self._joint_qkv(
            attn, hidden_states, encoder_hidden_states, image_rotary_emb, attention_mask is not None
        )
        if joint_qkv is None:
            joint_qkv = self._reference_joint_qkv(
                attn, hidden_states, encoder_hidden_states, image_rotary_emb
            )
        joint_query, joint_key, joint_value = joint_qkv

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


@maybe_allow_in_graph
class QwenImageTransformerBlock(nn.Module):
    _inference_kernels: ModuleType | None = None
    _fuse_linear_gelu: bool = True

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.zero_cond_t = zero_cond_t

    def _modulate(self, x, mod_params, index=None):
        """Apply modulation to input tensor"""
        # x: b l d, shift: b d, scale: b d, gate: b d
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result

    def _inference_mlp(self, mlp: FeedForward, x: torch.Tensor) -> torch.Tensor:
        if not self._fuse_linear_gelu:
            return mlp(x)
        activation = mlp.net[0]
        if (
            type(activation) is GELU
            and activation.approximate == "tanh"
            and not mlp._forward_pre_hooks
            and not mlp._forward_hooks
            and not activation._forward_pre_hooks
            and not activation._forward_hooks
            and _plain_linear(activation.proj, x)
        ):
            x = self._inference_kernels.linear_gelu(x, activation.proj.weight, activation.proj.bias)
            return mlp.net[2](mlp.net[1](x))
        return mlp(x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        modulate_index: list[int] | None = None,
        modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modulation is None:
            img_mod_params = self.img_mod(temb)
            text_temb = temb.chunk(2, dim=0)[0] if self.zero_cond_t else temb
            txt_mod_params = self.txt_mod(text_temb)
        else:
            img_mod_params, txt_mod_params = modulation

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        kernels = self._inference_kernels
        fused = (
            kernels is not None
            and not self.training
            and modulate_index is None
            and isinstance(self.attn.processor, QwenDoubleStreamAttnProcessor2_0)
            and _use_inference_kernels(hidden_states)
            and _use_inference_kernels(encoder_hidden_states)
            and hidden_states.numel() > 0
            and encoder_hidden_states.numel() > 0
        )
        if fused:
            img_modulated = kernels.norm_modulate(hidden_states, img_mod1, self.img_norm1.eps)
            txt_modulated = kernels.norm_modulate(encoder_hidden_states, txt_mod1, self.txt_norm1.eps)
        else:
            img_modulated, img_gate1 = self._modulate(
                self.img_norm1(hidden_states), img_mod1, modulate_index
            )
            txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        joint_attention_kwargs = joint_attention_kwargs or {}
        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if fused:
            hidden_states, img_modulated2 = kernels.residual_norm_modulate(
                hidden_states, img_attn_output, img_mod1, img_mod2, self.img_norm2.eps
            )
            encoder_hidden_states, txt_modulated2 = kernels.residual_norm_modulate(
                encoder_hidden_states, txt_attn_output, txt_mod1, txt_mod2, self.txt_norm2.eps
            )
            img_mlp_output = self._inference_mlp(self.img_mlp, img_modulated2)
            hidden_states = kernels.gated_residual(hidden_states, img_mlp_output, img_mod2)
            txt_mlp_output = self._inference_mlp(self.txt_mlp, txt_modulated2)
            encoder_hidden_states = kernels.gated_residual(encoder_hidden_states, txt_mlp_output, txt_mod2)
        else:
            hidden_states = hidden_states + img_gate1 * img_attn_output
            encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
            img_modulated2, img_gate2 = self._modulate(
                self.img_norm2(hidden_states), img_mod2, modulate_index
            )
            hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)
            txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
            encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QwenImageTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    """
    The Transformer model introduced in Qwen.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImageTransformerBlock"]
    # Make CP plan compatible with https://github.com/huggingface/diffusers/pull/12702
    _cp_plan = {
        "transformer_blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "transformer_blocks.*": {
            "modulate_index": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
            "encoder_hidden_states_mask": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        "pos_embed": {
            0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    # Tensor-parallel plan: how each block's fused/unfused Linears shard across the TP mesh. Qwen-Image is fully
    # unfused (separate Q/K/V and separate img/txt MLPs), so every entry is plain "colwise"/"rowwise" (torch's
    # ColwiseParallel / RowwiseParallel) and no packed sharding is needed. AdaLN modulation (img_mod/txt_mod) and
    # norm_out stay replicated (intentionally absent here).
    _tp_plan = {
        # double-stream (MMDiT) blocks
        "transformer_blocks.*.attn.to_q": "colwise",
        "transformer_blocks.*.attn.to_k": "colwise",
        "transformer_blocks.*.attn.to_v": "colwise",
        "transformer_blocks.*.attn.to_out.0": "rowwise",
        "transformer_blocks.*.attn.add_q_proj": "colwise",
        "transformer_blocks.*.attn.add_k_proj": "colwise",
        "transformer_blocks.*.attn.add_v_proj": "colwise",
        "transformer_blocks.*.attn.to_add_out": "rowwise",
        "transformer_blocks.*.img_mlp.net.0.proj": "colwise",
        "transformer_blocks.*.img_mlp.net.2": "rowwise",
        "transformer_blocks.*.txt_mlp.net.0.proj": "colwise",
        "transformer_blocks.*.txt_mlp.net.2": "rowwise",
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        if not use_layer3d_rope:
            self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        else:
            self.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, use_additional_t_cond=use_additional_t_cond
        )

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.zero_cond_t = zero_cond_t

    def enable_inference_optimizations(self, *, fuse_linear_gelu: bool = True) -> None:
        """Prepare BF16 CUDA inference for the current device and weights."""
        # Load Triton when the caller enables CUDA inference optimizations.
        from . import inference_kernels

        for block in self.transformer_blocks:
            block._inference_kernels = inference_kernels
            block._fuse_linear_gelu = fuse_linear_gelu
            processor = block.attn.processor
            if isinstance(processor, QwenDoubleStreamAttnProcessor2_0):
                processor._inference_kernels = inference_kernels

    @torch.no_grad()
    def forward_cfg(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        negative_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        img_shapes: list[tuple[int, int, int]] | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        negative_encoder_hidden_states_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        additional_t_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive and negative predictions from serial CFG at the same timestep."""
        assert not self.training, "forward_cfg requires eval mode"
        positive_image = self.img_in(hidden_states)
        temb, modulate_index = self._timestep_conditioning(
            timestep, positive_image, img_shapes, None, additional_t_cond
        )
        # Keep modulation local to each CFG pair so the next call uses its own timestep.
        text_temb = temb.chunk(2, dim=0)[0] if self.zero_cond_t else temb
        modulation = tuple(
            (block.img_mod(temb), block.txt_mod(text_temb)) for block in self.transformer_blocks
        )

        text_len = encoder_hidden_states.shape[1]
        if image_rotary_emb is None:
            image_rotary_emb = self.pos_embed(
                img_shapes, max_txt_seq_len=text_len, device=positive_image.device
            )

        negative_text_len = negative_encoder_hidden_states.shape[1]
        negative_rotary_emb = image_rotary_emb
        if negative_text_len != text_len:
            assert img_shapes is not None, "img_shapes required for different text lengths"
            negative_rotary_emb = self.pos_embed(
                img_shapes, max_txt_seq_len=negative_text_len, device=positive_image.device
            )

        positive = self._forward_transformer(
            positive_image, encoder_hidden_states, encoder_hidden_states_mask,
            temb, modulate_index, image_rotary_emb, None, None, modulation,
        )
        negative = self._forward_transformer(
            self.img_in(hidden_states), negative_encoder_hidden_states, negative_encoder_hidden_states_mask,
            temb, modulate_index, negative_rotary_emb, None, None, modulation,
        )
        return positive, negative

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask for the encoder hidden states. Expected to have 1.0 for valid tokens and 0.0 for padding tokens.
                Used in the attention processor to prevent attending to padding tokens. The mask can have any pattern
                (not just contiguous valid tokens followed by padding) since it's applied element-wise in attention.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            img_shapes (`list[tuple[int, int, int]]`, *optional*):
                Image shapes for RoPE computation.
            guidance (`torch.Tensor`, *optional*):
                Guidance tensor for conditional generation.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_block_samples (*optional*):
                ControlNet block samples to add to the transformer blocks.
            additional_t_cond (`torch.Tensor`, *optional*):
                Additional timestep conditioning added to the timestep embedding.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        hidden_states = self.img_in(hidden_states)
        temb, modulate_index = self._timestep_conditioning(
            timestep, hidden_states, img_shapes, guidance, additional_t_cond
        )
        if image_rotary_emb is None:
            image_rotary_emb = self.pos_embed(
                img_shapes, max_txt_seq_len=encoder_hidden_states.shape[1], device=hidden_states.device
            )
        output = self._forward_transformer(
            hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb,
            modulate_index, image_rotary_emb, attention_kwargs, controlnet_block_samples,
        )
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def _timestep_conditioning(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        img_shapes: list[tuple[int, int, int]] | None,
        guidance: torch.Tensor | None,
        additional_t_cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        return temb, modulate_index

    def _forward_transformer(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None,
        temb: torch.Tensor,
        modulate_index: torch.Tensor | None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_kwargs: dict[str, Any] | None,
        controlnet_block_samples: list[torch.Tensor] | None,
        modulation: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        _, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                    attention_kwargs,
                    modulate_index,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                    modulate_index=modulate_index,
                    modulation=None if modulation is None else modulation[index_block],
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)
