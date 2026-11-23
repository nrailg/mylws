# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Standalone grouped SwiGLU from G-Core's Qwen4ExpEPExperts._local_experts."""

import torch
from torch import Tensor
from torch.nn.functional import silu
from transformers.integrations.moe import _grouped_linear


def grouped_swiglu(rows: Tensor, offsets: Tensor, gate_up_proj: Tensor, down_proj: Tensor) -> Tensor:
    """Apply each expert's two Linear projections to its consecutive rows."""
    if rows.shape[0] == 0:
        # Keep both weight gradients defined when no tokens reach this rank.
        return rows + gate_up_proj[:0].sum() + down_proj[:0].sum()

    gate_up = _grouped_linear(rows, gate_up_proj, offsets, bias=None, is_transposed=False)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = silu(gate) * up
    return _grouped_linear(activated, down_proj, offsets, bias=None, is_transposed=False)


def eager_experts(rows: Tensor, expert_ids: Tensor, gate_up_proj: Tensor, down_proj: Tensor) -> Tensor:
    sorted_ids, perm = torch.sort(expert_ids)
    counts = torch.bincount(sorted_ids, minlength=gate_up_proj.size(0))
    offsets = counts.cumsum(dim=0, dtype=torch.int32)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device)
    return grouped_swiglu(rows[perm], offsets, gate_up_proj, down_proj)[inverse]


def prepare_deepep_experts(ids: Tensor, counts: list[int]) -> tuple[Tensor, Tensor]:
    flat_ids = ids.flatten()
    valid = torch.nonzero_static(flat_ids >= 0, size=sum(counts)).flatten()
    sorted_ids, perm = torch.sort(flat_ids.index_select(0, valid))
    offsets = torch.searchsorted(
        sorted_ids, torch.arange(len(counts), device=ids.device), right=True, out_int32=True,
    )
    return valid.index_select(0, perm), offsets


def deepep_experts(
    rows: Tensor, layout: tuple[Tensor, Tensor], weights: Tensor, gate_up_proj: Tensor,
    down_proj: Tensor, *, fused_slots: bool,
) -> Tensor:
    slots, offsets = layout
    if fused_slots:
        from .moe_slots import gather_slots, weighted_reduce_slots

        expanded, inverse = gather_slots(rows, slots, weights.size(1))
        projected = grouped_swiglu(expanded, offsets, gate_up_proj, down_proj)
        return weighted_reduce_slots(projected, weights, slots, inverse)

    row_ids = slots // weights.size(1)
    expanded = rows.index_select(0, row_ids)
    projected = grouped_swiglu(expanded, offsets, gate_up_proj, down_proj)
    weighted = (projected * weights.flatten().index_select(0, slots)[:, None]).to(rows.dtype)
    return torch.zeros_like(rows).index_add_(0, row_ids, weighted)
