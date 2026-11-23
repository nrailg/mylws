# Lightweight libs

每个子目录放一个独立的 lib，将 `src` 加入 `PYTHONPATH` 后使用。

## my_gemm

`from my_gemm import my_gemm`，调用 `C = my_gemm(A, B)` 计算 `A @ B`。输入为同一 CUDA 设备上的 BF16 矩阵，形状分别为 `[M, K]` 和 `[K, N]`，输出为 `[M, N]`。三个维度均须大于零，支持非连续输入和尾块。

TileLang NN kernel 用 `T.dynamic` 声明 `M`、`N`、`K`，不同形状复用同一份编译结果。forward 计算 `A @ B`，backward 按需计算 `dC @ B.T` 和 `A.T @ dC`，均用 FP32 累加并返回 BF16。每次调用先将操作数转为连续矩阵，再断言首地址和行步长均按 16 字节对齐。对 BF16，forward 要求 `K` 和 `N` 是 8 的倍数，计算 `dB` 时还要求 `M` 是 8 的倍数。

### my_grouped_mm

`my_grouped_mm(mat_a, mat_b, *, offs)` 实现 MoE 训练用到的两种 grouped matrix multiply。输入与输出均为 CUDA BF16，kernel 使用 FP32 累加，两种模式都支持一阶 autograd。[PyTorch grouped_mm 接口](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.grouped_mm.html)

| 模式 | 每个 expert 的乘法，`s:e` 为该组的 token 行 | 输出形状 |
| --- | --- | --- |
| 2D × 3D | `A[s:e] @ B[g]`，其中 `A[T,K]`、`B[E,K,N]` | `[T,N]` |
| 2D × 2D | `A[:,s:e] @ B[s:e,:]`，其中 `A[M,T_a]`、`B[T_b,N]` | `[E,M,N]` |

`offs` 是同一设备上长度为 `E` 的连续 `int32` tensor，记录每组的累计结束位置。例如 `[3, 3, 8]` 分别分配 3、0、5 行。最后一个 offset 可以等于分组维度的长度，也可以只覆盖前缀。2D × 3D 的未用输出行及其输入梯度为零，2D × 2D 的空组结果为零。入口检查 offset 的范围和顺序时会同步一次 CUDA。

Qwen3.8-Flash-Next 的 expert 权重 `W` 按 `[E, D_o, D_i]` 存储。`gate_up_proj` 为 `[512, 1280, 2560]`，`down_proj` 为 `[512, 2560, 640]`。传入 `W.mT` 时 kernel 直接读取原权重的存储。kernel 使用的 BF16 矩阵首地址、物理行步长及三维操作数的 expert 步长均须按 16 字节对齐，各 expert 的 token 数可以为奇数。[模型定义](my_chunked_ep/model/qwen4_exp/modeling_qwen4_exp.py#L940-L972) · [Checkpoint 配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/config.json)

```python
from my_gemm import my_grouped_mm

y = my_grouped_mm(x, W.mT, offs=offs)  # [T, D_o]
y.backward(dy)
# dX = my_grouped_mm(dy, W, offs=offs)
# dW = my_grouped_mm(dy.T, x, offs=offs)
```

### 验证

依赖 Python 3.12、PyTorch、TileLang 和 CUDA GPU。测试额外使用 pytest，在仓库根目录运行以下命令。

```bash
PYTHONPATH=src python -m pytest -q tests/my_gemm
```

2026-11-23 在单张 RTX PRO 5000 Blackwell（SM120）上通过 9 项测试，含原有 4 项 GEMM 测试。5 项 grouped_mm 测试将 `D_i`、`D_o` 取为 2 的幂。Grouped Linear 用逐 expert 的 CPU FP32 Linear 核对输出、`dX` 和 `dW`，覆盖不等长分组、空组和未用尾部。权重梯度用例单独核对每组的 `dY.T @ X` 及该调用的一阶梯度。两种模式的输出和两侧梯度也与 CUDA 上的原生 `torch.nn.functional.grouped_mm` 在 BF16 容差内一致。

验证环境为 PyTorch `2.11.0+cu128`、TileLang `0.1.9`、apache-tvm-ffi `0.1.8.post2`。pytest 及原生算子对拍日志位于 `$MY_DFS/work/mylws/outputs/my_gemm/20261123-grouped-mm-pow2.se3D4n/verification.log`。

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
