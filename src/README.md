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

本地 Qwen4-Exp 源码来自 Transformers v5.24.0。[上游发布源码](https://github.com/huggingface/transformers/tree/v5.24.0/src/transformers/models/qwen4_exp)

`bench.py` 分别从本地目录和已安装的 `transformers==5.24.0` 导入模型与配置。两边各构造一层含 indexed attention 和 top-2 MoE 的 `Qwen4ExpForCausalLM`，将官方模型的随机权重完整载入本地模型，用同一批 token 做前向。下面的命令在单张 CUDA GPU 上运行。

```bash
python src/my_chunked_ep/model/qwen4_exp/bench.py
```

2026-11-22 在 H20、PyTorch 2.11.0+cu128 上通过。输出形状为 `[2, 8, 128]`，将每个 BF16 logit 的原始 16 位作为 `int16` 对比，2,048 个 logit 中差异为 0。[对拍代码](my_chunked_ep/model/qwen4_exp/bench.py)
