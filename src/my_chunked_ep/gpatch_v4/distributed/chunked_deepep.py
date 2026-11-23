# Copyright (c) 2026 Tencent Inc. All rights reserved.
from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from .chunked_ep import _chunk_sizes, _ExpertInput


class _ChunkedDeepEP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        rows: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        num_experts: int,
        buffer: Any,
        expert_fn: Callable,
        needs_grad: bool,
        prepare_fn: Callable | None,
        num_chunks: int,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        sizes = _chunk_sizes(rows.shape[0], num_chunks)
        pending, keepalive = [], []
        for x, index, prob in zip(rows.split(sizes), ids.split(sizes), weights.split(sizes)):
            if index.numel() == 0:
                # DeepEP compares raw pointers; empty views can retain nonempty storage.
                index, prob = index.clone(), prob.clone()
            layout = buffer.get_dispatch_layout(
                index,
                num_experts,
                previous_event=buffer.capture(),
                async_finish=True,
            )
            per_rank, per_rdma_rank, per_expert, in_rank, event = layout
            keepalive.append((layout, index, prob))
            pending.append(
                buffer.dispatch(
                    x,
                    topk_idx=index,
                    topk_weights=prob,
                    num_tokens_per_rank=per_rank,
                    num_tokens_per_rdma_rank=per_rdma_rank,
                    num_tokens_per_expert=per_expert,
                    is_token_in_rank=in_rank,
                    previous_event=event,
                    async_finish=True,
                )
            )

        saved, graphs, handles, returned, saved_ranges = [], [], [], [], []
        context = weakref.ref(ctx)

        def pack(tensor: torch.Tensor) -> int:
            saved.append(tensor)
            return len(saved) - 1

        def unpack(index: int) -> torch.Tensor:
            return context().active_saved[index]

        ep_chunk_count = len(sizes)
        for i in range(ep_chunk_count):
            local_rows, local_ids, local_weights, counts, handle, event = pending[i]
            event.current_stream_wait()
            layout = local_ids if prepare_fn is None else prepare_fn(local_ids, counts)
            saved_start = len(saved)
            with torch.profiler.record_function("chunked_deepep::forward_experts"), torch.autocast(
                device_type="cuda",
                enabled=False,
            ):
                if needs_grad:
                    # Bridge saved tensors through the outer checkpoint/offload hooks.
                    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(
                        pack, unpack
                    ):
                        anchor = rows.new_empty(0, requires_grad=True)
                        leaves = (
                            _ExpertInput.apply(local_rows.detach(), anchor),
                            _ExpertInput.apply(local_weights.detach(), anchor),
                            *(
                                _ExpertInput.apply(p.detach(), anchor)
                                if p.requires_grad else p.detach() for p in params
                            ),
                        )
                        local_output = expert_fn(leaves[0], layout, leaves[1], *leaves[2:])
                        edges = tuple(
                            torch.autograd.graph.get_gradient_edge(p)
                            for p in leaves if p.requires_grad
                        )
                        graphs.append((torch.autograd.graph.get_gradient_edge(local_output), edges))
                else:
                    local_output = expert_fn(local_rows, layout, local_weights, *params)
            saved_ranges.append((saved_start, len(saved)))
            returned.append(
                buffer.combine(
                    local_output.detach(),
                    handle,
                    previous_event=buffer.capture(),
                    async_finish=True,
                )
            )
            keepalive.append(local_output)
            handles.append(handle)
            pending[i] = None

        outputs = []
        for output, _, event in returned:
            event.current_stream_wait()
            outputs.append(output)
        # Empty expert graphs must also trigger collective checkpoint replay on every rank.
        ctx.save_for_backward(*saved, rows.new_empty(0))
        saved.clear()
        ctx.graphs, ctx.handles, ctx.buffer = graphs, handles, buffer
        ctx.saved_ranges = saved_ranges
        ctx.sizes, ctx.top_k = sizes, weights.shape[1]
        ctx.param_meta = [(p.shape, p.dtype, p.device, p.requires_grad) for p in params]
        return torch.cat(outputs, dim=0)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        keep_graph = torch._C._autograd._get_current_graph_task_keep_graph()
        ctx.active_saved = list(ctx.saved_tensors)
        if not keep_graph:
            ctx.maybe_clear_saved_tensors()
        buffer = ctx.buffer
        grad_chunks = [grad.contiguous() for grad in grad_output.split(ctx.sizes)]
        pending = [
            buffer.dispatch(
                grad.contiguous(),
                handle=handle,
                previous_event=buffer.capture(),
                async_finish=True
            ) for grad, handle in zip(grad_chunks, ctx.handles)
        ]
        returned, keepalive, param_grads = [], [], [None] * len(ctx.param_meta)
        ep_chunk_count = len(ctx.sizes)
        for i in range(ep_chunk_count):
            local_grad, _, _, _, _, event = pending[i]
            event.current_stream_wait()
            output_edge, edges = ctx.graphs[i]
            with torch.profiler.record_function("chunked_deepep::backward_experts"):
                grads = torch.autograd.grad(
                    output_edge, edges, local_grad, allow_unused=True, retain_graph=keep_graph
                )
            dx, dw = grads[:2]
            if dx is None:
                dx = torch.zeros_like(local_grad)
            if dw is None:
                dw = local_grad.new_zeros((local_grad.shape[0], ctx.top_k), dtype=torch.float32)
            dx, dw = dx.contiguous(), dw.float().contiguous()
            keepalive.append((dx, dw))
            returned.append(
                buffer.combine(
                    dx,
                    ctx.handles[i],
                    topk_weights=dw,
                    previous_event=buffer.capture(),
                    async_finish=True,
                )
            )
            if not keep_graph:
                start, end = ctx.saved_ranges[i]
                ctx.active_saved[start:end] = [None] * (end - start)
            active_grads = iter(grads[2:])
            for j, (_, _, _, requires_grad) in enumerate(ctx.param_meta):
                if requires_grad:
                    grad = next(active_grads)
                    if grad is not None:
                        if param_grads[j] is None:
                            # A returned gradient may alias dX or have expanded strides.
                            param_grads[j] = grad.clone()
                        else:
                            param_grads[j].add_(grad)
            pending[i] = None
            del grads, active_grads
            grad = None

        dxs, dws = [], []
        for dx, dw, event in returned:
            event.current_stream_wait()
            dxs.append(dx)
            dws.append(dw)
        for j, (shape, dtype, device, requires_grad) in enumerate(ctx.param_meta):
            if requires_grad and param_grads[j] is None:
                param_grads[j] = torch.zeros(shape, dtype=dtype, device=device)
        del ctx.active_saved
        return torch.cat(dxs
                        ), None, torch.cat(dws), None, None, None, None, None, None, *param_grads


def chunked_deepep(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    parameters: tuple[torch.Tensor, ...],
    expert_fn: Callable[..., torch.Tensor],
    num_experts: int,
    buffer: Any,
    *,
    prepare_fn: Callable | None = None,
    num_chunks: int = 2,
) -> torch.Tensor:
    """Pipeline token chunks using ordinary DeepEP dispatch and combine.

    Parameters
    ----------
    hidden_states : torch.Tensor
        BF16 CUDA token rows of shape ``[N, H]`` supported by the DeepEP buffer.
    expert_indices, routing_weights : torch.Tensor
        Shape ``[N, K]``; int64 global expert ids and differentiable router weights.
    parameters : tuple of torch.Tensor
        Explicit local expert parameters; gradients return through outer autograd/FSDP2.
    expert_fn : Callable
        Pure ``fn(recv_rows, local_topk_ids, recv_weights, *parameters)`` returning
        weighted outputs in the deduplicated receive-row layout. Must handle empty
        inputs and preserve a gradient edge. No RNG, collectives or parameter hooks.
    num_experts : int
    buffer : deep_ep.Buffer
        Ordinary DeepEP buffer with capture/dispatch/combine APIs. All ranks must
        issue calls in the same order with matching grad mode, including empty chunks.
    prepare_fn : Callable, optional
        ``fn(local_topk_ids, tokens_per_expert)`` returning routing metadata passed
        to ``expert_fn`` instead of ids. Counts are DeepEP's CPU list without padding.
    num_chunks : int, default 2
        At least two. All chunks are dispatched before local expert computation;
        combine results remain live until concatenation.

    Returns
    -------
    torch.Tensor
        Shape ``[N, H]``. Supports first derivatives and checkpoint/offload. All
        chunks use the buffer's communication stream; waits occur only on consumption.

    Raises
    ------
    ValueError
        If token/routing shapes, devices or dtypes do not match the contract.
    """
    if (
        hidden_states.ndim != 2 or expert_indices.ndim != 2 or
        routing_weights.shape != expert_indices.shape or
        expert_indices.shape[0] != hidden_states.shape[0] or expert_indices.shape[1] == 0
    ):
        raise ValueError("chunked_deepep expects hidden [N,H] and indices/weights [N,K]")
    if (
        not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16 or
        expert_indices.device != hidden_states.device or
        routing_weights.device != hidden_states.device or expert_indices.dtype != torch.int64 or
        not routing_weights.is_floating_point() or num_experts <= 0
    ):
        raise ValueError(
            "chunked_deepep requires BF16 CUDA rows, int64 indices and floating weights"
        )
    return _ChunkedDeepEP.apply(
        hidden_states.contiguous(),
        expert_indices.contiguous(),
        routing_weights.float().contiguous(),
        num_experts,
        buffer,
        expert_fn,
        torch.is_grad_enabled() and (
            hidden_states.requires_grad or routing_weights.requires_grad or
            any(p.requires_grad for p in parameters)
        ),
        prepare_fn,
        num_chunks,
        *parameters,
    )
