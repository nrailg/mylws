# Lightweight libs

每个子目录放一个独立的 lib，将 `src` 加入 `PYTHONPATH` 后使用。

## my_fsdp

`MinimalFSDP(block)` 沿第 0 维对每个 dense FP32 可训练参数单独分片。创建 wrapper 时，传入的 block 会注册本 rank 的 shard，所以应在包装之后创建 optimizer。各 rank 使用同样的初始权重，执行相同的 block 调用和梯度路径，各自处理本地 batch。

以下代码在每个 rank 上运行，假设默认 process group 和本 rank 的 `device` 已设置。

```python
import torch
from torch import nn
from my_fsdp import MinimalFSDP

torch.manual_seed(123)
block = MinimalFSDP(nn.Linear(5, 5).to(device))
optimizer = torch.optim.AdamW(block.parameters())
torch.manual_seed(456 + torch.distributed.get_rank())
h = torch.randn(3, 5, device=device)
for _ in range(3):
    h = block(h)
h.square().mean().backward()
optimizer.step()
```

每次调用用 `all_gather_into_tensor` 重建权重，交给 `functional_call` 执行 block，返回时释放完整权重的 storage。进入该次调用的 backward 时，再次 all-gather 到同一块 storage，使 autograd 保存的 weight view 可读。block 的反向计算结束后，每个有梯度的参数执行一次 `reduce_scatter_tensor(SUM)` 并除以 `world_size`。奇数行产生的不等长 shard 只在通信时补齐。[PyTorch FSDP2 关于 storage 和 autograd 的设计说明](https://github.com/pytorch/pytorch/issues/114299)

每个 wrapper 管理一个独立的参数集合，输入和输出各为一个 Tensor。多次调用同一 wrapper 可共享权重和 optimizer state。在同一计算图的 backward 完成后再更新 shard。block 应只读完整权重，对需要梯度的输入使用 out-of-place 运算。像 `Embedding(max_norm=...)` 这样的原地权重更新会触发断言。`Embedding(sparse=True)` 的梯度会转为 dense 再通信。此版本使用同步 collective 和一阶 autograd，跨 wrapper 的 tied weights、mixed precision、通信合并、prefetch 和 checkpoint 存取不在范围内。

### 验证

测试自行启动两个 rank。两者输入不同，分别与 DDP 对照 1 轮和 3 轮调用的输出、输入梯度、参数梯度及两次 AdamW 更新和动量。同时检查通信次数、forward 调用次数、完整权重的 storage 大小，以及 Embedding 的稀疏梯度和原地修改限制。

```bash
# macOS，两个 CPU 进程
GLOO_SOCKET_IFNAME=lo0 PYTHONPATH=src python -m pytest -q tests/my_fsdp
# Linux，同一节点上的两张 CUDA GPU
CUDA_VISIBLE_DEVICES=0,1 MY_FSDP_TEST_BACKEND=nccl PYTHONPATH=src python -m pytest -q tests/my_fsdp
```

2026-11-23 在 macOS（PyTorch 2.15.0，Gloo）及双 RTX PRO 5000 Blackwell（PyTorch 2.11.0+cu128，NCCL）上均通过 3 项测试。GPU 日志位于 `$MY_DFS/work/mylws/outputs/my_fsdp/20261123-full-shard.iFDt85/pytest.log`。

## my_gemm

`from my_gemm import my_gemm`，调用 `C = my_gemm(A, B)` 计算 `A @ B`。输入为同一 CUDA 设备上的 BF16 矩阵，形状分别为 `[M, K]` 和 `[K, N]`，输出为 `[M, N]`。三个维度均须大于零，支持非连续输入和尾块。

TileLang NN kernel 用 `T.dynamic` 声明 `M`、`N`、`K`，不同形状复用同一份编译结果。forward 计算 `A @ B`，backward 按需计算 `dC @ B.T` 和 `A.T @ dC`，均用 FP32 累加并返回 BF16。每次调用先将操作数转为连续矩阵，再断言首地址和行步长均按 16 字节对齐。对 BF16，forward 要求 `K` 和 `N` 是 8 的倍数，计算 `dB` 时还要求 `M` 是 8 的倍数。

### my_grouped_mm

`my_grouped_mm(mat_a, mat_b, *, offs)` 接受按 expert 排列的输入 `[T, D_i]` 和右操作数 `[E, D_i, D_o]`。对 expert `g` 的 token 行 `s:e`，forward 计算 `mat_a[s:e] @ mat_b[g]`，输出形状为 `[T, D_o]`。输入与输出均为 CUDA BF16，kernel 使用 FP32 累加，支持一阶 autograd。TileLang 用 `T.empty` 分配输出。[PyTorch grouped_mm 接口](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.grouped_mm.html) · [TileLang grouped GEMM 示例](https://github.com/tile-ai/tilelang/blob/3f01780f093cc5f6691d106da287a0eeb8e51ad3/examples/grouped_gemm/example_grouped_gemm_bwd.py)

`offs` 是同一设备上长度为 `E` 的连续 `int32` tensor，记录每组的累计结束位置。全部输入行都属于某个 expert，要求 `offs[-1] == T`。例如 `[2, 2, 6, 8]` 给 4 个 expert 分别分配 2、0、4、2 行。空组的权重梯度为零。

前向和 `dX` 沿输出的 M、N 两维切成 `64 × 64` 的 tile。前向在 GPU 上从 `offs` 算出各 expert 的行数、起点和对齐后的起点，`dX` 复用这些 metadata。kernel 按对齐后的起点找到 M tile 所属的 expert。JIT 按特征宽度编译，token 数和 expert 数在运行时传入。

Qwen3.8-Flash-Next 的 expert 权重 `W` 按 `[E, D_o, D_i]` 存储。`gate_up_proj` 为 `[512, 1280, 2560]`，`down_proj` 为 `[512, 2560, 640]`。[模型定义](my_chunked_ep/model/qwen4_exp/modeling_qwen4_exp.py#L940-L972) · [Checkpoint 配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/config.json)

调用方传入 `B = W.mT` 时，入口用 `.contiguous()` 将这个转置视图复制一次。求 `dX` 时复用这份连续的 `B`，用 `T.gemm` 的 `transpose_B=True` 计算 `dY @ B.T`。权重梯度 kernel 在各 expert 的 token 归约尾块补零，直接输出 `dB = X.T @ dY`，PyTorch autograd 再将梯度传给 `W`。

```python
from my_gemm import my_grouped_mm

y = my_grouped_mm(x, W.mT, offs=offs)  # [T, D_o]
y.backward(dy)  # x.grad [T, D_i]，W.grad [E, D_o, D_i]
```

### 验证

依赖 Python 3.12、PyTorch、TileLang 和 CUDA GPU。测试额外使用 pytest，在仓库根目录运行以下命令。

```bash
PYTHONPATH=src python -m pytest -q tests/my_gemm
```

2026-11-23 在单张 RTX PRO 5000 Blackwell（SM120）上，修正归约尾块的 mask 后，移植版和简化版各通过 8 项测试。用逐 expert 的 CPU FP32 Linear 核对输出、`dX` 和 `dW`，另测了特征尾块、首尾空组以及特征宽度相同时的不同 token 数。

验证环境为 PyTorch `2.11.0+cu128`、TileLang `0.1.9`、apache-tvm-ffi `0.1.8.post2`。日志位于 `$MY_DFS/work/mylws/outputs/my_gemm/20261123-grouped-port.wMXPBH/simplified.log`。

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
