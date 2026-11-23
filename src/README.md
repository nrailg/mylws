# Lightweight libs

每个子目录放一个独立的 lib，将 `src` 加入 `PYTHONPATH` 后使用。

## my_gemm

`from my_gemm import my_gemm`，调用 `C = my_gemm(A, B)` 计算 `A @ B`。输入为同一 CUDA 设备上的 BF16 矩阵，形状分别为 `[M, K]` 和 `[K, N]`，输出为 `[M, N]`。三个维度均须大于零，支持非连续输入和尾块。

TileLang NN kernel 用 `T.dynamic` 声明 `M`、`N`、`K`，不同形状复用同一份编译结果。forward 计算 `A @ B`，backward 按需计算 `dC @ B.T` 和 `A.T @ dC`，均用 FP32 累加并返回 BF16。每次调用先将操作数转为连续矩阵，再断言首地址和行步长均按 16 字节对齐。对 BF16，forward 要求 `K` 和 `N` 是 8 的倍数，计算 `dB` 时还要求 `M` 是 8 的倍数。

依赖 Python 3.12、PyTorch、TileLang 和 CUDA GPU。测试额外使用 pytest，在仓库根目录运行：

```bash
PYTHONPATH=src python -m pytest -q tests/my_gemm
```

2026-11-23 在单张 RTX PRO 5000 Blackwell（SM120）上通过 4 项测试。整块 `(M,N,K) = (256,384,128)` 和尾块 `(136,192,80)` 均用 PyTorch FP32 autograd 核对 forward 输出与两个输入的梯度。另两项分别验证首地址和行步长未对齐时触发断言。验证环境使用 PyTorch `2.11.0+cu128`、TileLang `0.1.9`、apache-tvm-ffi `0.1.8.post2`。日志位于 `$MY_DFS/work/mylws/outputs/my_gemm/20261123/pytest-dynamic-nn-aligned.log`。

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

G-Core 的 chunkedEP 使用 `Qwen/Qwen3.8-Flash-Next`。本地 Qwen4-Exp 实现和 Transformers v5.24.0 分别加载 BF16 checkpoint，记录加载耗时并逐位比较 forward logits。[运行命令与实验记录](my_chunked_ep/README.md)
