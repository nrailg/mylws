# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Chunked EP with asynchronous exchanges and streaming weighted reduction."""
from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

ExpertFn = Callable[..., torch.Tensor]


def _chunk_sizes(num_tokens: int, num_chunks: int) -> tuple[int, ...]:
    if isinstance(num_chunks, bool) or not isinstance(num_chunks, int) or num_chunks < 2:
        raise ValueError("num_chunks must be an integer >= 2")
    size, remainder = divmod(num_tokens, num_chunks)
    return tuple(size + (i < remainder) for i in range(num_chunks))


@dataclass
class _Route:
    order: torch.Tensor
    inverse: torch.Tensor
    send: list[int]
    recv: list[int]
    ids: torch.Tensor


@dataclass
class _Exchange:
    source: torch.Tensor
    result: torch.Tensor
    work: dist.Work

    def wait(self) -> torch.Tensor:
        self.work.wait()
        return self.result


def _exchange(
    source: torch.Tensor,
    send: list[int],
    recv: list[int],
    group: dist.ProcessGroup,
) -> _Exchange:
    source = source.contiguous()
    result = source.new_empty((sum(recv), *source.shape[1:]))
    work = dist.all_to_all_single(
        result,
        source,
        output_split_sizes=recv,
        input_split_sizes=send,
        group=group,
        async_op=True,
    )
    return _Exchange(source, result, work)


def _routes(
    ids: tuple[torch.Tensor, ...],
    num_local_experts: int,
    group: dist.ProcessGroup,
) -> list[_Route]:
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    orders = [torch.argsort(chunk // num_local_experts, stable=True) for chunk in ids]
    counts = [torch.bincount(chunk // num_local_experts, minlength=size) for chunk in ids]
    pending = [_exchange(count, [1] * size, [1] * size, group) for count in counts]
    splits = [(count.tolist(), item.wait().tolist()) for count, item in zip(counts, pending)]
    pending_ids = [
        _exchange(chunk[order], send, recv, group)
        for chunk, order, (send, recv) in zip(ids, orders, splits)
    ]
    routes = []
    for order, (send, recv), item in zip(orders, splits, pending_ids):
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        routes.append(_Route(order, inverse, send, recv, item.wait() - rank * num_local_experts))
    return routes


class _ExpertInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        # A gradient edge without an AccumulateGrad leaf retaining the input storage.
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[None, None]:
        return None, None


class _ChunkedExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rows: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        sizes: tuple[int, ...],
        num_local_experts: int,
        group: dist.ProcessGroup,
        expert_fn: ExpertFn,
        needs_grad: bool,
        prepare_fn: Callable | None,
        fused_reduce: bool,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        top_k = weights.shape[1]
        offsets = tuple(accumulate((0, *sizes)))
        routes = _routes(ids.split(tuple(size * top_k for size in sizes)), num_local_experts, group)
        layouts = [route.ids if prepare_fn is None else prepare_fn(route.ids) for route in routes]
        pending, combined = [None] * len(routes), [None] * len(routes)
        graphs, reduce_graphs, saved = [], [], []
        saved_ranges = [[] for _ in routes]
        output = rows.new_empty(rows.shape)
        context = weakref.ref(ctx)

        def pack(tensor: torch.Tensor) -> int:
            saved.append(tensor)
            return len(saved) - 1

        def unpack(index: int) -> torch.Tensor:
            return context().active_saved[index]

        def dispatch(i: int) -> _Exchange:
            route = routes[i]
            return _exchange(rows[route.order // top_k + offsets[i]], route.send, route.recv, group)

        def reduce(i: int) -> None:
            projected = combined[i].wait().index_select(0, routes[i].inverse).to(rows.dtype)
            combined[i] = None
            probs = weights[offsets[i]:offsets[i + 1]]
            saved_start = len(saved)
            with torch.set_grad_enabled(needs_grad), torch.autograd.graph.saved_tensors_hooks(
                pack, unpack
            ):
                if needs_grad:
                    anchor = rows.new_empty(0, requires_grad=True)
                    projected = _ExpertInput.apply(projected.detach(), anchor)
                    probs = (
                        _ExpertInput.apply(probs.detach(), anchor)
                        if weights.requires_grad else probs.detach()
                    )
                if (
                    fused_reduce and projected.is_cuda and
                    projected.dtype in (torch.bfloat16, torch.float32) and
                    probs.dtype == torch.float32
                ):
                    from gpatch_v4.kernel.triton.moe_slots import (
                        weighted_reduce_contiguous,
                    )

                    reduced = weighted_reduce_contiguous(projected, probs.contiguous())
                else:
                    reduced = _WeightedReduce.apply(projected, probs, (sizes[i] + 1) // 2)
                if needs_grad:
                    edges = tuple(
                        torch.autograd.graph.get_gradient_edge(p)
                        for p in (projected, probs) if p.requires_grad
                    )
                    reduce_graphs.append((torch.autograd.graph.get_gradient_edge(reduced), edges))
            output[offsets[i]:offsets[i + 1]].copy_(reduced.detach())
            saved_ranges[i].append((saved_start, len(saved)))

        pending[0] = dispatch(0)
        for i, route in enumerate(routes):
            if i + 1 < len(routes):
                pending[i + 1] = dispatch(i + 1)
            local_rows = pending[i].wait()
            pending[i] = None
            saved_start = len(saved)
            with torch.profiler.record_function("chunked_ep::forward_experts"), torch.autocast(
                device_type=rows.device.type,
                enabled=False,
            ):
                if needs_grad:
                    # Bridge nested saved tensors to the outer checkpoint/offload hooks.
                    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(
                        pack, unpack
                    ):
                        anchor = rows.new_empty(0, requires_grad=True)
                        leaves = (
                            _ExpertInput.apply(local_rows.detach(), anchor), *(
                                _ExpertInput.apply(p.detach(), anchor)
                                if p.requires_grad else p.detach() for p in params
                            )
                        )
                        local_output = expert_fn(leaves[0], layouts[i], *leaves[1:])
                        edges = tuple(
                            torch.autograd.graph.get_gradient_edge(p)
                            for p in leaves if p.requires_grad
                        )
                        graphs.append((torch.autograd.graph.get_gradient_edge(local_output), edges))
                else:
                    local_output = expert_fn(local_rows, layouts[i], *params)
            saved_ranges[i].append((saved_start, len(saved)))
            combined[i] = _exchange(local_output.detach(), route.recv, route.send, group)
            del local_rows, local_output
            if i:
                reduce(i - 1)
        reduce(len(routes) - 1)
        ctx.saved_count = len(saved)
        # Routing tensors must participate in outer checkpoint/offload hooks too.
        for route in routes:
            saved.extend((route.order, route.inverse, route.ids))
        ctx.save_for_backward(*saved, rows.new_empty(0))
        saved.clear()
        ctx.graphs = graphs
        ctx.reduce_graphs = reduce_graphs
        ctx.saved_ranges = saved_ranges
        ctx.param_meta = [(p.shape, p.dtype, p.device, p.requires_grad) for p in params]
        ctx.route_splits = [(route.send, route.recv) for route in routes]
        ctx.group = group
        ctx.sizes, ctx.offsets, ctx.top_k = sizes, offsets, top_k
        ctx.weight_meta = (weights.shape, weights.dtype, weights.device, weights.requires_grad)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        keep_graph = torch._C._autograd._get_current_graph_task_keep_graph()
        saved = ctx.saved_tensors
        ctx.active_saved = list(saved[:ctx.saved_count])
        route_tensors = saved[ctx.saved_count:-1]
        routes = [
            _Route(
                route_tensors[3 * i], route_tensors[3 * i + 1], send, recv, route_tensors[3 * i + 2]
            ) for i, (send, recv) in enumerate(ctx.route_splits)
        ]
        del saved, route_tensors
        if not keep_graph:
            ctx.maybe_clear_saved_tensors()
        pending, returned = [None] * len(routes), [None] * len(routes)
        param_grads = [None] * len(ctx.param_meta)
        grad_rows = torch.empty_like(grad_output)
        shape, dtype, device, train_weights = ctx.weight_meta
        grad_weights = torch.empty(shape, dtype=dtype, device=device) if train_weights else None

        def dispatch(i: int) -> _Exchange:
            output_edge, edges = ctx.reduce_graphs[i]
            grads = torch.autograd.grad(
                output_edge,
                edges,
                grad_output[ctx.offsets[i]:ctx.offsets[i + 1]],
                retain_graph=keep_graph,
            )
            if train_weights:
                grad_weights[ctx.offsets[i]:ctx.offsets[i + 1]].copy_(grads[1])
            route = routes[i]
            return _exchange(grads[0][route.order], route.send, route.recv, ctx.group)

        def consume(i: int) -> None:
            chunk = returned[i].wait().index_select(0, routes[i].inverse)
            returned[i] = None
            torch.sum(
                chunk.view(ctx.sizes[i], ctx.top_k, grad_output.shape[1]),
                dim=1,
                out=grad_rows[ctx.offsets[i]:ctx.offsets[i + 1]],
            )

        pending[0] = dispatch(0)
        for i, route in enumerate(routes):
            if i + 1 < len(routes):
                pending[i + 1] = dispatch(i + 1)
            local_grad = pending[i].wait()
            pending[i] = None
            output_edge, edges = ctx.graphs[i]
            with torch.profiler.record_function("chunked_ep::backward_experts"):
                grads = torch.autograd.grad(
                    output_edge, edges, local_grad, allow_unused=True, retain_graph=keep_graph
                )
            dx = grads[0]
            if dx is None:
                dx = torch.zeros_like(local_grad)
            returned[i] = _exchange(dx, route.recv, route.send, ctx.group)
            if not keep_graph:
                for start, end in ctx.saved_ranges[i]:
                    ctx.active_saved[start:end] = [None] * (end - start)
                ctx.graphs[i] = ctx.reduce_graphs[i] = None
            active_grads = iter(grads[1:])
            for j, (_, _, _, requires_grad) in enumerate(ctx.param_meta):
                if not requires_grad:
                    continue
                grad = next(active_grads)
                if grad is not None:
                    if param_grads[j] is None:
                        # Autograd can return expanded gradients or storage shared with dX.
                        param_grads[j] = grad.clone()
                    else:
                        # Preserve parameter-dtype accumulation across chunks.
                        param_grads[j].add_(grad)
            del grads, local_grad, active_grads, dx
            grad = None
            if i:
                consume(i - 1)
        consume(len(routes) - 1)
        for j, (shape, dtype, device, requires_grad) in enumerate(ctx.param_meta):
            if requires_grad and param_grads[j] is None:
                param_grads[j] = torch.zeros(shape, dtype=dtype, device=device)
        del ctx.active_saved
        return grad_rows, None, grad_weights, None, None, None, None, None, None, None, *param_grads


class _WeightedReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output: torch.Tensor, weights: torch.Tensor, split: int) -> torch.Tensor:
        n, top_k = weights.shape
        rows = output.view(n, top_k, output.shape[-1])
        result = output.new_empty((n, output.shape[-1]))
        for start, end in ((0, split), (split, n)):
            weighted = (rows[start:end] * weights[start:end, :, None]).to(output.dtype)
            torch.sum(weighted, dim=1, out=result[start:end])
            del weighted
        ctx.save_for_backward(
            output if ctx.needs_input_grad[1] else None,
            weights if ctx.needs_input_grad[0] else None
        )
        ctx.shape = (n, top_k, output.shape[-1])
        ctx.output_dtype, ctx.weight_dtype = output.dtype, weights.dtype
        ctx.split = split
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        output, weights = ctx.saved_tensors
        n, top_k, hidden = ctx.shape
        rows = None if output is None else output.view(n, top_k, hidden)
        grad_rows = grad.new_empty(ctx.shape,
                                   dtype=ctx.output_dtype) if ctx.needs_input_grad[0] else None
        grad_weights = grad.new_empty((n, top_k),
                                      dtype=ctx.weight_dtype) if ctx.needs_input_grad[1] else None
        dtype = torch.promote_types(ctx.output_dtype, ctx.weight_dtype)
        for start, end in ((0, ctx.split), (ctx.split, n)):
            incoming = grad[start:end, None, :].to(dtype)
            if grad_rows is not None:
                grad_rows[start:end] = incoming * weights[start:end, :, None]
            if grad_weights is not None:
                grad_weights[start:end] = (incoming * rows[start:end]).sum(dim=-1)
        grad_output = None if grad_rows is None else grad_rows.view(n * top_k, hidden)
        return grad_output, grad_weights, None


def chunked_ep(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    parameters: tuple[torch.Tensor, ...],
    expert_fn: ExpertFn,
    num_local_experts: int,
    group: dist.ProcessGroup,
    prepare_fn: Callable[[torch.Tensor], Any] | None = None,
    *,
    fused_reduce: bool = False,
    num_chunks: int = 2,
) -> torch.Tensor:
    """Pipeline EP chunks and reduce each returned chunk on its token owner.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape ``[N, H]``; CUDA/NCCL for overlap, CPU/Gloo for correctness tests.
    expert_indices, routing_weights : torch.Tensor
        Shape ``[N, K]``, int64 global expert ids and differentiable router weights.
        Compute the router once before calling this function.
    parameters : tuple of torch.Tensor
        All local expert parameters, passed explicitly for standard autograd/FSDP2.
    expert_fn : Callable
        Pure, deterministic ``fn(rows, local_ids, *parameters) -> rows``. Must handle
        empty rows and retain a differentiable output. No RNG, collectives, parameter
        hooks or mutable state inside this callback. Autocast is disabled for expert
        computation; parameter dtypes control precision in forward and backward.
    num_local_experts : int
    group : ProcessGroup
        All ranks must call forward/backward in the same order, with matching grad mode.
    prepare_fn : Callable, optional
        Pure local-id layout preparation, run before payload exchanges. When provided,
        its result replaces ``local_ids`` in ``expert_fn``.
    fused_reduce : bool, default False
        Fuse output weighting/reduction and its backward for BF16/FP32 CUDA rows
        and FP32 weights; requires Triton. CPU and other dtypes keep the eager path.
    num_chunks : int, default 2
        At least two. Every rank issues this many chunks, including empty chunks.
        Payload prefetch and unconsumed combine results are bounded to two chunks.

    Returns
    -------
    torch.Tensor
        Shape ``[N, H]``. Saved expert graphs support first derivatives and activation
        checkpoint/offload; normal backward does not recompute expert forward.
        Count/id exchange is a synchronous setup phase. Payload collectives use
        asynchronous NCCL work: dispatch chunk 1 can overlap compute chunk 0, and
        combine chunk 0 can overlap compute chunk 1. Throughput requires measurement.

    Raises
    ------
    ValueError
        If routing shapes or dtypes do not match the contract.
    """
    if hidden_states.ndim != 2 or expert_indices.ndim != 2 or routing_weights.shape != expert_indices.shape:
        raise ValueError("chunked_ep expects hidden [N,H] and indices/weights [N,K]")
    if expert_indices.shape[0] != hidden_states.shape[0] or expert_indices.shape[1] == 0:
        raise ValueError("chunked_ep requires matching token counts and positive top-k")
    if expert_indices.dtype != torch.int64 or not routing_weights.is_floating_point():
        raise ValueError("chunked_ep requires int64 indices and floating-point routing weights")
    if num_local_experts <= 0:
        raise ValueError("num_local_experts must be positive")
    output = _ChunkedExperts.apply(
        hidden_states,
        expert_indices.reshape(-1),
        routing_weights,
        _chunk_sizes(hidden_states.shape[0], num_chunks),
        num_local_experts,
        group,
        expert_fn,
        torch.is_grad_enabled() and (
            hidden_states.requires_grad or routing_weights.requires_grad or
            any(p.requires_grad for p in parameters)
        ),
        prepare_fn,
        fused_reduce,
        *parameters,
    )
    return output
