import torch
import torch.distributed as dist
from flash_attn_interface import flash_attn_func


class AllGatherAttention:
    """固定 shape 的 BSHD forward 示例；各 rank 持有连续的 sequence shard。"""

    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        heads_per_group: int,
        causal: bool,
        group: dist.ProcessGroup,
    ) -> None:
        batch, local_seq, q_heads, head_dim = q.shape
        kv_heads = k.shape[2]
        assert q.is_cuda and k.device == q.device
        assert q.dtype == k.dtype == torch.bfloat16
        assert head_dim == 128 and q_heads > 0 and kv_heads > 0
        assert k.shape == (batch, local_seq, kv_heads, head_dim)
        assert q_heads % kv_heads == 0
        assert 0 < heads_per_group <= kv_heads and kv_heads % heads_per_group == 0
        assert torch.cuda.get_device_capability(q.device)[0] == 9

        self.q_shape = q.shape
        self.k_shape = k.shape
        self.group = group
        self.heads_per_group = heads_per_group
        self.q_heads_per_group = heads_per_group * (q_heads // kv_heads)
        self.num_groups = kv_heads // heads_per_group
        self.causal = causal
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        self.kv_end = (rank + 1) * local_seq if causal else world_size * local_seq
        self.num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        self.comm_stream = torch.cuda.Stream(device=q.device)
        # sequence 放在最前面，使 AllGather 直接按 rank 拼接每个 batch 的 sequence。
        self.buffers = [
            torch.empty(
                (world_size * local_seq, 2, batch, heads_per_group, head_dim),
                dtype=k.dtype,
                device=k.device,
            )
            for _ in range(min(2, self.num_groups))
        ]
        self.ready = [torch.cuda.Event() for _ in range(2)]

    def _gather(self, packed: torch.Tensor, index: int) -> None:
        slot = index % 2
        # 同时等待输入 pack 和上一次使用该 buffer 的 FA3 完成。
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            with torch.profiler.record_function(f"allgather_head_group_{index}"):
                work = dist.all_gather_into_tensor(
                    self.buffers[slot], packed[index], group=self.group, async_op=True
                )
                # Work.wait 让当前 CUDA stream 等待 NCCL，不做 device synchronize。
                work.wait()
                self.ready[slot].record()

    @torch.no_grad()
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        overlap: bool,
        sm_margin: int = 0,
    ) -> torch.Tensor:
        assert q.shape == self.q_shape and k.shape == v.shape == self.k_shape
        assert q.device == k.device == v.device == self.buffers[0].device
        assert q.dtype == k.dtype == v.dtype == self.buffers[0].dtype
        assert not (q.requires_grad or k.requires_grad or v.requires_grad)
        assert 0 <= sm_margin < self.num_sms
        batch, local_seq, _, head_dim = k.shape
        # 一次 pack 所有 head group，避免每轮 packing 阻塞下一次通信的提交。
        packed = (
            torch.stack((k, v), dim=2)
            .reshape(batch, local_seq, 2, self.num_groups, self.heads_per_group, head_dim)
            .permute(3, 1, 2, 0, 4, 5)
            .contiguous()
        )
        compute_stream = torch.cuda.current_stream()
        outputs = []
        self._gather(packed, 0)
        for index in range(self.num_groups):
            compute_stream.wait_event(self.ready[index % 2])
            if overlap and index + 1 < self.num_groups:
                self._gather(packed, index + 1)

            kv = self.buffers[index % 2][: self.kv_end]
            k_group = kv[:, 0].permute(1, 0, 2, 3)
            v_group = kv[:, 1].permute(1, 0, 2, 3)
            q_start = index * self.q_heads_per_group
            q_group = q[:, :, q_start : q_start + self.q_heads_per_group]
            with torch.profiler.record_function(f"attention_head_group_{index}"):
                outputs.append(
                    flash_attn_func(
                        q_group,
                        k_group,
                        v_group,
                        causal=self.causal,
                        num_splits=1,
                        sm_margin=sm_margin,
                    )
                )
            if not overlap and index + 1 < self.num_groups:
                self._gather(packed, index + 1)

        # 最后一次 ready 的等待已将全部通信 join 回 compute stream。
        packed.record_stream(self.comm_stream)
        return torch.cat(outputs, dim=2)
