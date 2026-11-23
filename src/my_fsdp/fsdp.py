from typing import override

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import FunctionCtx, once_differentiable
from torch.func import functional_call


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    if tensor.size(0) == rows:
        return tensor.contiguous()
    padded = tensor.new_zeros((rows, *tensor.shape[1:]))
    padded[: tensor.size(0)].copy_(tensor)
    return padded


class _GatheredParameter:
    """一次 block 调用中，一个参数的通信 buffer 及其本地 shard。"""

    def __init__(self, shard: nn.Parameter, rows: int) -> None:
        self.shard = shard
        self.rows = rows
        self.world_size = dist.get_world_size()
        self.chunk_rows = (rows + self.world_size - 1) // self.world_size
        self.buffer = shard.new_empty((self.world_size * self.chunk_rows, *shard.shape[1:]))

    @torch.no_grad()
    def all_gather(self) -> None:
        self.buffer.untyped_storage().resize_(self.buffer.nbytes)
        dist.all_gather_into_tensor(self.buffer, _pad_rows(self.shard, self.chunk_rows))

    def weight(self) -> torch.Tensor:
        # 共享 storage，独立使用 version counter。反向回填 buffer 时保存的 weight view 仍然有效。
        shape = (self.rows, *self.shard.shape[1:])
        with torch.inference_mode(False):  # 推理时同样检查 block 是否原地修改权重。
            return self.buffer.new_empty(0).set_(self.buffer.untyped_storage(), 0, shape)

    def reshard(self) -> None:
        self.buffer.untyped_storage().resize_(0)

    def reduce_scatter(self, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.to_dense()  # 例如 Embedding(sparse=True) 会返回 COO gradient。
        reduced = grad.new_empty((self.chunk_rows, *grad.shape[1:]))
        padded = _pad_rows(grad, self.buffer.size(0))
        dist.reduce_scatter_tensor(reduced, padded, op=dist.ReduceOp.SUM)
        return reduced[: self.shard.size(0)] / self.world_size


class _Unshard(torch.autograd.Function):
    @staticmethod
    @override
    def forward(
        ctx: FunctionCtx, x: torch.Tensor, params: list[_GatheredParameter], *shards: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        ctx.params = params
        ctx.set_materialize_grads(False)
        if not x.requires_grad:
            ctx.mark_non_differentiable(x)
        for param in params:
            param.all_gather()
        # x 也经过此节点，保证 block 对输入和权重的反向计算都完成后才释放 storage。
        return (x, *(param.weight() for param in params))

    @staticmethod
    @override
    @once_differentiable
    def backward(
        ctx: FunctionCtx, grad_x: torch.Tensor | None, *weight_grads: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        shard_grads = []
        for param, grad in zip(ctx.params, weight_grads, strict=True):
            shard_grads.append(param.reduce_scatter(grad) if grad is not None else None)
            param.reshard()
        return (grad_x, None, *shard_grads)


class MinimalFSDP(nn.Module):
    """将 block 的 dense FP32 参数沿第 0 维分片，使用默认 process group 训练。"""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        rank, world_size = dist.get_rank(), dist.get_world_size()
        self.block = block
        self._rows: dict[str, int] = {}
        local_shards: dict[nn.Parameter, nn.Parameter] = {}
        for name, weight in block.named_parameters():
            assert weight.layout == torch.strided and weight.dtype == torch.float32
            assert weight.requires_grad and weight.ndim >= 1
            rows = weight.size(0)
            chunk_rows = (rows + world_size - 1) // world_size
            start = rank * chunk_rows
            local = weight.detach()[start : start + chunk_rows]
            local_shards[weight] = nn.Parameter(local.clone(memory_format=torch.contiguous_format))
            self._rows[name] = rows
        for name, weight in list(block.named_parameters(remove_duplicate=False)):
            parent, _, attr = name.rpartition(".")
            block.get_submodule(parent).register_parameter(attr, local_shards[weight])

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shards = dict(self.block.named_parameters())
        params = [_GatheredParameter(shard, self._rows[name]) for name, shard in shards.items()]
        # 显式传入 shard 叶子，使 autograd 将 reduce-scatter 的结果累加到 shard.grad。
        x, *weights = _Unshard.apply(x, params, *shards.values())
        versions = [weight._version for weight in weights]
        try:
            output = functional_call(self.block, dict(zip(shards, weights, strict=True)), (x,))
            current_versions = [weight._version for weight in weights]
            assert versions == current_versions, "block modified a gathered weight"
            assert isinstance(output, torch.Tensor)
            # 返回 weight 的 view 时，输出需要在释放临时 storage 后仍可读取。
            if any(output.untyped_storage() is param.buffer.untyped_storage() for param in params):
                output = output.clone()
        finally:
            for param in params:
                param.reshard()

        if output.requires_grad:
            def restore_weights(_grad: torch.Tensor) -> None:
                for param in params:
                    param.all_gather()

            output.register_hook(restore_weights)
        return output
