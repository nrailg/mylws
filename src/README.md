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

本地 Qwen4-Exp 源码来自 Transformers v5.24.0。G-Core 的 chunkedEP 对应 `Qwen/Qwen3.8-Flash-Next`。对拍使用该 checkpoint 的文本模型 `Qwen4ExpForCausalLM`，共 48 层，每层从 512 个专家中选取 10 个。[模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) · [上游源码](https://github.com/huggingface/transformers/tree/v5.24.0/src/transformers/models/qwen4_exp)

`bench.py` 的 `load_checkpoint` 分别用本地实现和官方实现从同一目录加载 BF16 权重。两次前向使用相同的设备分布和 token，依次运行以复用显存，最后逐位比较全部 logits。运行需要 CUDA、PyTorch、`transformers==5.24.0`、Accelerate 及完整的本地 Hugging Face checkpoint。

```bash
python src/my_chunked_ep/model/qwen4_exp/bench.py /path/to/Qwen3.8-Flash-Next
```

2026-11-23 在一台 8 卡 H20 节点上验证，使用 PyTorch 2.11.0+cu128 和 Accelerate 1.16.0。checkpoint 的配置和权重索引与公开版本 `812d96e` 一致。attention 和 MoE 使用 eager，causal convolution 和 gated delta rule 使用 PyTorch reference。输入形状为 `[2, 8]`，输出形状为 `[2, 8, 248320]`，3,973,120 个 BF16 logits 全部逐位一致。[checkpoint 版本](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/812d96e10a0dd82b69d749e68693698cf6c36daa)

GPU 测试日志位于 `$MY_DFS/work/mylws/outputs/my_chunked_ep/20261123-qwen38-bf16.DoqX9u/forward.log`。
