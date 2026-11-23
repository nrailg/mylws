# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""G-Core Qwen4-Exp 的 eager EP，使用独立的 expert callback。"""

from collections.abc import Callable

import torch
import torch.distributed as dist


# 从 G-Core 0faff03f4 的 gpatch_v4/models/qwen4_exp/a2a.py 原样复制。
class _AllToAllUneven(torch.autograd.Function):
    """Uneven all-to-all along dim 0 with explicit per-rank split sizes.

    Forward
    -------
    Sends ``input[sum(in_splits[:i]) : sum(in_splits[:i+1])]`` to rank ``i`` for every
    ``i``. Receives from rank ``i`` a slice of length ``out_splits[i]``; concatenates
    received slices in rank order to form the output.

    Backward
    --------
    The transpose of an uneven all-to-all swaps ``in_splits`` and ``out_splits``
    (sender becomes receiver and vice versa), so backward applies the same op with the
    splits swapped.
    """
    @staticmethod
    def forward(ctx, group, input, in_splits, out_splits):
        ctx.group = group
        ctx.in_splits = in_splits
        ctx.out_splits = out_splits
        out = torch.empty(sum(out_splits), *input.shape[1:], dtype=input.dtype, device=input.device)
        dist.all_to_all_single(
            out,
            input.contiguous(),
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=group
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        return None, _AllToAllUneven.apply(
            ctx.group, grad, ctx.out_splits, ctx.in_splits
        ), None, None


def all_to_all_uneven(
    x: torch.Tensor,
    in_splits: list[int],
    out_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Uneven all-to-all along dim 0.

    Parameters
    ----------
    x : torch.Tensor
        ``x.shape[0]`` must equal ``sum(in_splits)``.
    in_splits : list[int]
        Dim-0 slice sizes to send to each rank.
    out_splits : list[int]
        Dim-0 slice sizes this rank receives from each rank.
        Must satisfy: ``in_splits[j]`` on rank ``i`` == ``out_splits[i]`` on rank ``j``.
    group : dist.ProcessGroup

    Returns
    -------
    torch.Tensor
        Shape ``[sum(out_splits), *x.shape[1:]]``.
    """
    return _AllToAllUneven.apply(group, x, in_splits, out_splits)


def eager_ep(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    parameters: tuple[torch.Tensor, ...],
    expert_fn: Callable[..., torch.Tensor],
    num_experts: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """从 G-Core 的 `Qwen4ExpEPExperts._forward_eager` 提取默认路径。

    每个 rank 持有连续的 expert ID，`expert_fn` 接收 dispatch 后的 rows 和本地 expert ID。
    """
    ep_size = group.size()
    assert num_experts > 0 and num_experts % ep_size == 0
    num_local_experts = num_experts // ep_size
    num_tokens, hidden_dim = hidden_states.shape
    top_k = expert_indices.shape[-1]

    # Each (token, slot) pair is an independent dispatch unit.
    flat_expert = expert_indices.reshape(-1)
    flat_weight = routing_weights.reshape(-1)
    flat_hidden = hidden_states.repeat_interleave(top_k, dim=0)

    # Group by owning rank so each rank's payload is contiguous.
    target_rank = torch.div(flat_expert, num_local_experts, rounding_mode="floor")
    order = torch.argsort(target_rank, stable=True)
    send_counts = torch.bincount(target_rank, minlength=ep_size)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()

    recv_hidden = all_to_all_uneven(flat_hidden[order], send_splits, recv_splits, group)
    recv_expert = all_to_all_uneven(flat_expert[order], send_splits, recv_splits, group)
    local_expert_id = recv_expert - group.rank() * num_local_experts
    local_output = expert_fn(recv_hidden, local_expert_id, *parameters)

    combined = all_to_all_uneven(local_output, recv_splits, send_splits, group)
    inverse_order = torch.empty_like(order)
    inverse_order[order] = torch.arange(order.numel(), device=order.device)
    combined = combined[inverse_order]
    combined = (combined * flat_weight.unsqueeze(-1)).to(hidden_states.dtype)
    return combined.view(num_tokens, top_k, hidden_dim).sum(dim=1)
