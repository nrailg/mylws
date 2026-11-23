"""Qwen4-Exp 的 eager EP、按行分片的 PLE lookup 和 Ulysses prefill。"""

from copy import copy
from math import lcm

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.tensor import Shard
from typing_extensions import override

if __package__:
    from . import modeling_qwen4_exp as qwen
else:
    import modeling_qwen4_exp as qwen


def split_counts(owners: Tensor, group: dist.ProcessGroup) -> tuple[list[int], list[int]]:
    send = torch.bincount(owners, minlength=group.size())
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return send.tolist(), recv.tolist()


def exchange_rows(x: Tensor, send: list[int], recv: list[int], group: dist.ProcessGroup) -> Tensor:
    out = x.new_empty((sum(recv), *x.shape[1:]))
    return dist_nn.all_to_all_single(out, x.contiguous(), recv, send, group)


class RowEmbedding(nn.Module):
    def __init__(self, rows: int, dim: int, group: dist.ProcessGroup, dtype: torch.dtype) -> None:
        super().__init__()
        self.group = group
        self.rows_per_rank = (rows + group.size() - 1) // group.size()
        count, self.start = Shard.local_shard_size_and_offset(rows, group.size(), group.rank())
        self.weight = nn.Parameter(torch.empty(count, dim, device="meta", dtype=dtype))

    @override
    def forward(self, ids: Tensor) -> Tensor:
        flat = ids.flatten()
        owners = flat // self.rows_per_rank
        order = owners.argsort(stable=True)
        send, recv = split_counts(owners, self.group)
        local_ids = exchange_rows(flat[order], send, recv, self.group) - self.start
        values = F.embedding(local_ids, self.weight)
        result = exchange_rows(values, recv, send, self.group)
        return result[order.argsort()].reshape(*ids.shape, self.weight.shape[1])


class ParallelExperts(qwen.Qwen4ExpTextExperts):
    def __init__(self, config: qwen.Qwen4ExpTextConfig, group: dist.ProcessGroup) -> None:
        assert config.num_experts % group.size() == 0
        expert_config = copy(config)
        expert_config.num_experts = config.num_experts // group.size()
        super().__init__(expert_config)
        self.group = group
        self.start = group.rank() * self.num_experts

    @override
    def forward(self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        slots = top_k_index.flatten()
        order = slots.argsort(stable=True)
        send, recv = split_counts(slots // self.num_experts, self.group)
        tokens = exchange_rows(hidden_states[order // top_k_index.size(1)], send, recv, self.group)
        experts = exchange_rows(slots[order], send, recv, self.group) - self.start
        if tokens.size(0):
            ones = tokens.new_ones((tokens.size(0), 1))
            values = super().forward(tokens, experts[:, None], ones)
        else:
            # 空接收端也要参与反向 all-to-all，且本地 expert 参数应得到零梯度。
            values = tokens + self.gate_up_proj[:0].sum() + self.down_proj[:0].sum()
        returned = exchange_rows(values, recv, send, self.group)
        by_slot = returned[order.argsort()].view(*top_k_index.shape, hidden_states.size(-1))
        weighted = by_slot * top_k_weights[..., None]
        # 与上游 eager 相同，每个 token 按 expert ID 顺序逐项做 BF16 累加。
        sorted_slots = top_k_index.argsort(dim=1)
        tokens = torch.arange(hidden_states.size(0), device=hidden_states.device)
        result = torch.zeros_like(hidden_states)
        for slot in sorted_slots.unbind(1):
            result = result + weighted[tokens, slot]
        return result


def exchange_axis(x: Tensor, scatter: int, gather: int, group: dist.ProcessGroup) -> Tensor:
    """在 [B, S, H, ...] 上交换序列与 head 分片，反向做逆交换。"""
    assert x.size(scatter) % group.size() == 0
    send = torch.stack(x.chunk(group.size(), dim=scatter))
    recv = dist_nn.all_to_all_single(torch.empty_like(send), send, group=group)
    return torch.cat(recv.unbind(), dim=gather)


def left_context(x: Tensor, width: int, group: dist.ProcessGroup) -> Tensor:
    tail = min(width, x.size(1))
    if tail == 0:
        return x
    ends = torch.cat(dist_nn.all_gather(x[:, -tail:].contiguous(), group=group), dim=1)
    stop = group.rank() * tail
    return torch.cat((ends[:, max(0, stop - width):stop], x), dim=1)


class UlyssesPLE(qwen.Qwen4ExpTextPLELayer):
    group: dist.ProcessGroup

    @override
    def forward(self, hidden_states: Tensor, input_ids: Tensor, cache=None, conv_mask=None) -> Tensor:
        assert cache is None and conv_mask is None
        # 卷积最早使用的前驱位置本身还需要 n-gram 的左侧 token。
        width = self.short_conv_state_len + self.ple_embedding.context_len
        states = left_context(hidden_states, width, self.group)
        ids = left_context(input_ids, width, self.group)
        return super().forward(states, ids, cache=None)[:, -input_ids.size(1):]


class UlyssesGatedDeltaNet(qwen.Qwen4ExpTextGatedDeltaNet):
    group: dist.ProcessGroup

    @override
    def forward(self, hidden_states: Tensor, cache=None, attention_mask=None, **kwargs) -> Tensor:
        assert cache is None and attention_mask is None
        group = self.group
        dims = (self.key_dim, self.key_dim, self.value_dim)
        head_dims = (self.head_k_dim, self.head_k_dim, self.head_v_dim)
        projected = self.in_proj_qkv(hidden_states).split(dims, dim=-1)
        full_seq = [
            exchange_axis(x.unflatten(-1, (-1, dim)), 2, 1, group).flatten(2)
            for x, dim in zip(projected, head_dims, strict=True)
        ]
        mixed = torch.cat(full_seq, dim=-1).transpose(1, 2)
        weights = self.conv1d.weight.squeeze(1).split(dims, dim=0)
        conv_weight = torch.cat([w.chunk(group.size())[group.rank()] for w in weights])
        mixed = qwen.causal_conv1d_fn(mixed, conv_weight, None, activation=self.activation)
        local_dims = [dim // group.size() for dim in dims]
        q, k, v = [
            x.unflatten(-1, (-1, dim))
            for x, dim in zip(mixed.transpose(1, 2).split(local_dims, -1), head_dims, strict=True)
        ]
        beta = exchange_axis(self.in_proj_b(hidden_states).sigmoid(), 2, 1, group)
        a = self.in_proj_a(hidden_states).float() + self.dt_bias
        decay = -self.A_log.float().exp() * F.softplus(a)
        decay = exchange_axis(decay, 2, 1, group)
        out, _ = qwen.torch_chunk_gated_delta_rule(
            q, k, v, g=decay, beta=beta, use_qk_l2norm_in_kernel=True
        )
        out = exchange_axis(out, 1, 2, group)
        z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
        out = self.norm(out.reshape_as(z), z).reshape(*hidden_states.shape[:2], self.value_dim)
        return self.out_proj(out)


class UlyssesAttention(qwen.Qwen4ExpTextAttention):
    group: dist.ProcessGroup

    @override
    def forward(
        self, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor, cache=None, **kwargs,
    ) -> tuple[Tensor, None]:
        assert cache is None
        group = self.group
        # indexer 只有一个 KV head。复用上游的全局选择，再按 head 并行计算主 attention。
        full_states = torch.cat(dist_nn.all_gather(hidden_states.contiguous(), group=group), dim=1)
        cos, sin = position_embeddings
        full_pos = tuple(
            torch.cat(dist_nn.all_gather(p.contiguous(), group=group), 1) for p in (cos, sin)
        )
        selected = self.indexer(full_states, full_pos, attention_mask, cache=None)
        mask = attention_mask.masked_fill(~selected, torch.finfo(attention_mask.dtype).min)

        batch, seq = hidden_states.shape[:2]
        shape = (batch, seq, -1, self.head_dim)
        q, gate = self.q_proj(hidden_states).view(batch, seq, -1, 2 * self.head_dim).chunk(2, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = qwen.apply_rotary_pos_emb(q, k, cos, sin)
        # 只在 KV head 无法均分时复制，剩余的 GQA 共享在通信后本地展开。
        transfer_heads = lcm(k.size(1), group.size())
        replicas = transfer_heads // k.size(1)
        local_groups = q.size(1) // transfer_heads
        k, v = qwen.repeat_kv(k, replicas), qwen.repeat_kv(v, replicas)
        q, k, v = [
            exchange_axis(x.transpose(1, 2), 2, 1, group).transpose(1, 2) for x in (q, k, v)
        ]
        k, v = qwen.repeat_kv(k, local_groups), qwen.repeat_kv(v, local_groups)
        scores = q @ k.transpose(-2, -1) * self.scaling + mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)
        out = (probs @ v).transpose(1, 2)
        out = exchange_axis(out, 1, 2, group).flatten(2) * gate.flatten(2).sigmoid()
        return self.o_proj(out), None
