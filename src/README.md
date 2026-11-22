# Lightweight libs

每个子目录放一个独立的 lib，将 `src` 加入 `PYTHONPATH` 后使用。

## myfa

从 gcore-dev 的 `gfused/myfa_sw_sinks.py` 和 `gfused/myfa_varlen_sw_sinks.py` 复制的 TileLang Flash Attention，包含 forward、backward、GQA、sliding window 和可选的 attention sinks。kernel 代码保持原样。

依赖 PyTorch、TileLang 和支持 TileLang 的 CUDA GPU。

```bash
PYTHONPATH=src python -c 'from myfa import myfa_sw_sinks, myfa_varlen_sw_sinks'
```

- `myfa_sw_sinks(q, k, v, ...)`：输入为 `[B, S, H, D]`，Q/K 序列长度须为 64 的倍数。
- `myfa_varlen_sw_sinks(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen, max_seqlen_k, ...)`：输入为 `[total_tokens, H, D]`，`max_seqlen_k` 须为 host int。
- `sinks=None`、`window_size=None` 可关闭对应功能；sliding window 仅支持 causal attention，具体限制沿用原实现。

正确性测试与实现放在同一目录，额外依赖 pytest；在 CUDA GPU 环境从仓库根目录运行：

```bash
PYTHONPATH=src python -m pytest src/myfa/test_myfa_sw_sinks.py src/myfa/test_myfa_varlen_sw_sinks.py -v
```

测试对比 PyTorch reference 的 forward 输出和 Q/K/V/sinks 梯度，未迁入原仓库的性能对比。

## my_fp8_quant

用 `y, scales = my_fp8_quant(x, round_scale=False)` 对有限的 CUDA BF16/FP32 输入按 128×128 分块量化。输入形状为 `[..., M, N]`，支持非连续输入，每个前导索引上的矩阵独立分块。边界块仅统计有效元素。

`y` 与输入同形状，dtype 为 E4M3FN。`scales` 是形状为 `[..., ceil(M/128), ceil(N/128)]` 的 FP32 tensor。例如，256×384 的矩阵返回 2×3 个 scale。每块的 scale 为 `max(absmax, 1e-4) / 448`，启用 `round_scale` 后向上取到最近的 2 的幂。反量化时将每个 FP8 元素转为 FP32 并乘以所属块的 scale。

依赖 PyTorch、TileLang 和 CUDA GPU。测试额外使用 pytest 和 DeepSeek TileKernels，在仓库根目录运行：

```bash
PYTHONPATH=src python -m pytest -q src/my_fp8_quant/test_my_fp8_quant.py
```

2026-11-18 在单张 H20 上通过 10 项测试。PyTorch reference 覆盖 batch、非连续输入和边界块，DeepSeek TileKernels 的 `per_block_cast` 用于核对 FP8 编码和两种 scale 模式。验证环境为 PyTorch 2.11.0+cu128、TileLang 0.1.9 和 TileKernels `a9dcaf0`。

GPU 测试日志位于 `$MY_DFS/work/mylws/outputs/my_fp8_quant/20261118/pytest-block128.log`。

## my_chunked_ep

基于 PyTorch all-to-all 的分块专家并行。每个 rank 持有连续的 `num_local_experts` 个专家；先对本地 token 做一次路由，再调用：

```python
from my_chunked_ep import chunked_ep

output = chunked_ep(hidden, ids, weights, (local_params,), expert_fn,
                    num_local_experts, group, num_chunks=2)
```

`hidden` 为浮点 `[N,H]`，`ids` 和 `weights` 为 `[N,K]` 的 int64 全局专家编号和浮点权重。各 rank 的 N 可以不同。各 rank 须使用相同的块数（至少 2）、梯度配置和 collective 调用顺序，包括空块。回调 `expert_fn(rows, local_ids, *parameters)` 接收本 rank 的行及本地专家编号，应支持空输入，返回同形状、同 dtype 的行。训练时使用确定性的可微回调，所有可训练参数显式传入。

每块按目标 rank 打包，提前 dispatch 下一块，在本地计算专家后异步 combine，回到来源 rank 恢复 slot 顺序、乘权重并求和。反向复用同一通信顺序，重算每块专家前向以求一阶梯度。最小实现保留各块返回值，未集成 DeepEP、融合 kernel 或 FSDP。

在两张 CUDA GPU 上运行输出和梯度对照：

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=2 tests/chunked_ep/check_chunked_ep.py
```

2026-11-22 在单机 2 张 H20（PyTorch 2.11.0+cu128、NCCL）通过 6 组数值对照和 1 项未使用专家参数零梯度回归。FP64 和 BF16 均覆盖不均匀 token 数、空 rank、空块及集中路由；检查输出、输入梯度、路由权重梯度和本地专家参数梯度。
