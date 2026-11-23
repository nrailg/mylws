import copy
import os
from collections.abc import Callable
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import override
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from my_fsdp import MinimalFSDP


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(5, 7)
        self.down = nn.Linear(7, 5)
        self.scale = nn.Parameter(torch.tensor([0.5]))  # rank 1 的 shard 为空

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.down(self.up(x).tanh())


def _run_rank(
    rank: int, backend: str, init_method: str, check: Callable[[int, torch.device], None]
) -> None:
    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    dist.init_process_group(
        backend, init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        check(rank, device)
    finally:
        dist.destroy_process_group()


def _compare_training(rank: int, device: torch.device, *, loops: int) -> None:
    torch.manual_seed(123)
    block = _Block().to(device)
    ref_block = copy.deepcopy(block)
    ref = DistributedDataParallel(nn.Sequential(*([ref_block] * loops)))
    model = MinimalFSDP(block)
    optim = torch.optim.AdamW(model.parameters(), lr=0.005)
    ref_optim = torch.optim.AdamW(ref.parameters(), lr=0.005)
    full_storages: list[torch.UntypedStorage] = []

    def capture_weights(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        for weight in module.parameters():
            storage = weight.untyped_storage()
            assert storage.nbytes() >= weight.nbytes > 0
            full_storages.append(storage)

    block.register_forward_hook(capture_weights)
    torch.manual_seed(456 + rank)
    for step in range(2):
        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        x = torch.randn(3, 5, device=device, requires_grad=(step == 0))
        ref_x = x.detach().clone().requires_grad_(x.requires_grad)
        target = torch.randn_like(x)
        full_storages.clear()

        ref_y = ref(ref_x)
        with (
            patch.object(dist, "all_gather_into_tensor", wraps=dist.all_gather_into_tensor)
            as all_gather,
            patch.object(dist, "reduce_scatter_tensor", wraps=dist.reduce_scatter_tensor)
            as reduce_scatter,
        ):
            y = x
            for _ in range(loops):
                y = model(y)
                assert all(storage.nbytes() == 0 for storage in full_storages)
            torch.testing.assert_close(y, ref_y)
            gathers_per_pass = loops * len(list(block.parameters()))
            assert all_gather.call_count == gathers_per_pass
            (y - target).square().mean().backward()
            assert all_gather.call_count == 2 * gathers_per_pass
            assert reduce_scatter.call_count == gathers_per_pass
        (ref_y - target).square().mean().backward()
        assert len(full_storages) == gathers_per_pass  # backward 没有重跑 block
        assert all(storage.nbytes() == 0 for storage in full_storages)
        torch.testing.assert_close(x.grad, ref_x.grad)

        for shard, weight in zip(block.parameters(), ref.parameters(), strict=True):
            reference_shard = torch.tensor_split(weight.grad, 2, dim=0)[rank]
            torch.testing.assert_close(shard.grad, reference_shard)
            assert shard.untyped_storage().nbytes() == shard.nbytes
            assert shard.grad.untyped_storage().nbytes() == shard.grad.nbytes

        optim.step()
        ref_optim.step()
        for shard, weight in zip(block.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(shard, torch.tensor_split(weight, 2, dim=0)[rank])
            for name in ("exp_avg", "exp_avg_sq"):
                moment = torch.tensor_split(ref_optim.state[weight][name], 2, dim=0)[rank]
                torch.testing.assert_close(optim.state[shard][name], moment)


def _check_embedding(rank: int, device: torch.device) -> None:
    torch.manual_seed(321)
    dense = nn.Embedding(5, 3).to(device)
    sparse = copy.deepcopy(dense)
    sparse.sparse = True
    ref = DistributedDataParallel(dense)
    model = MinimalFSDP(sparse)
    indices = torch.tensor([rank, rank + 1, rank + 1], device=device)
    model(indices).sum().backward()
    ref(indices).sum().backward()
    torch.testing.assert_close(sparse.weight.grad, torch.tensor_split(dense.weight.grad, 2)[rank])

    renorm = MinimalFSDP(nn.Embedding(5, 3, max_norm=0.1).to(device))
    for mode in (torch.enable_grad(), torch.inference_mode()):
        with mode, pytest.raises(AssertionError, match="block modified a gathered weight"):
            renorm(indices)


@pytest.mark.parametrize(
    "check",
    [partial(_compare_training, loops=1), partial(_compare_training, loops=3), _check_embedding],
    ids=["one-loop", "three-loops", "embedding"],
)
def test_full_shard(check: Callable[[int, torch.device], None], tmp_path: Path) -> None:
    """用双 rank 对照 DDP 的梯度及更新，验证 storage 生命周期和 Embedding 边界。"""
    backend = os.environ.get("MY_FSDP_TEST_BACKEND", "gloo")
    assert backend in ("gloo", "nccl")
    mp.spawn(_run_rank, (backend, (tmp_path / "store").as_uri(), check), nprocs=2)
