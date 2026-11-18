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

`my_fp8_quant(x, group_size=128, round_scale=False)` 沿最后一维分组，将有限的 CUDA BF16/FP32 输入转成 E4M3FN。支持任意前导维度和非连续输入，最后一维须能被 `group_size`（64 或 128）整除。

返回的 `(y, scales)` 中，`y` 与输入同形状，`scales` 的最后一维是组数。每组的 FP32 scale 为 `max(absmax, 1e-4) / 448`，启用 `round_scale` 后向上取到最近的 2 的幂。反量化时将每组 FP8 数据转为 FP32 并乘以对应的 scale。

依赖 PyTorch、TileLang 和 CUDA GPU。测试额外使用 pytest 和 DeepSeek TileKernels，在仓库根目录运行：

```bash
PYTHONPATH=src python -m pytest -q src/my_fp8_quant/test_my_fp8_quant.py
```

2026-11-18 在单张 NVIDIA H20 上通过全部 14 项测试，使用 PyTorch 2.11.0+cu128、TileLang 0.1.9、apache-tvm-ffi 0.1.9 和 TileKernels `a9dcaf0`。测试逐字节比较 FP8 输出；与 TileKernels 对比时，普通 scale 允许相差 1 ULP，2 的幂 scale 要求完全一致。

GPU 测试日志位于 `$MY_DFS/work/mylws/outputs/my_fp8_quant/20261118/pytest.log`。
