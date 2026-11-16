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
