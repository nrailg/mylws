# Qwen3.8-Flash-Next checkpoint 加载与 forward 对拍

对拍使用 G-Core 对应的 `Qwen/Qwen3.8-Flash-Next` 文本模型 `Qwen4ExpForCausalLM`，共 48 层，每层从 512 个专家中选取 10 个。checkpoint 的配置和权重索引与公开版本 `812d96e` 一致。[checkpoint 版本](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/812d96e10a0dd82b69d749e68693698cf6c36daa) · [Transformers v5.24.0 源码](https://github.com/huggingface/transformers/tree/v5.24.0/src/transformers/models/qwen4_exp)

## 运行

运行需要 CUDA、PyTorch、Transformers 5.24.0、Accelerate 和完整的 checkpoint。在单个 launcher 的仓库根目录运行，默认读取节点上的 `/hf-hub/Qwen/Qwen3.8-Flash-Next`。日常对拍保留前 4 层，覆盖第 1～3 层的 GatedDeltaNet 和第 4 层带 QSA indexer 的 indexed attention。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1
python archived/model/qwen4_exp/bench.py --debug-num-layers 4
```

`--debug-num-layers N` 保留前 N 层及输入、输出权重。第 2 层含约 95 GiB 的 PLE embedding，单层调试会跳过这张表。Transformers 的加载报告会列出被裁掉层的参数，bench 仍会检查所有保留层。省略开关即加载全部 48 层。要复现 CephFS 对照，将 `QWEN_CHECKPOINT` 设为该模型在 CephFS 上的目录。

```bash
python archived/model/qwen4_exp/bench.py "$QWEN_CHECKPOINT"
```

本地模型先加载 BF16 权重并自动推断 device map，`pub_qwen` 随后从同一目录加载。跨设备 dispatch 时复用第一份模型的 device map。如果映射只包含一个计算设备，Transformers 5.24.0 省略 `hf_device_map`，第二次加载也使用 `auto`。每次计时从调用 `load_checkpoint` 开始，到持有模型权重的 GPU 完成同步为止。两次 forward 使用相同的 token，逐位比较全部 logits。[bench.py](qwen4_exp/bench.py)

## 单机 8 卡分布式对拍

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
torchrun --standalone --nproc-per-node=8 archived/model/qwen4_exp/bench.py \
  --debug-num-layers 4 --fsdp --ep-size 4 --cp-size 2
```

三个开关可以分别启用。FSDP2 将 dense 参数分片到全部 rank。启用 EP 后，每组按 expert ID 分配 routed experts，持有同一组 experts 的 rank 再做 expert FSDP。Router 和 shared expert 仍走 dense 路径。各 rank 在 meta 上建模，只从 safetensors 读取自己的参数切片。PLE 表始终按词表行分布，通过 all-to-all 做 lookup，FSDP 跳过这张表。[分片与加载](qwen4_exp/parallel_model.py)

CP 组内切分同一条等长、无 padding 的 causal prefill 序列，组间使用不同样本。Ulysses 将局部 token 的 Q/K/V 交换为完整序列的局部 heads，再做逆交换。GatedDeltaNet 的卷积和递推使用完整序列。QSA 的 indexer 在 CP 组内复用全局输入并做全局 top-k。PLE 从左侧 rank 取得 n-gram 与卷积历史。序列长度、query head 数、GatedDeltaNet 的 key/value head 数须能被 CP 度数整除。[EP、PLE 与 Ulysses](qwen4_exp/parallel_layers.py)

rank 0 在分布式模型释放后独立加载官方 Transformers，比较拼回原顺序的 CPU logits。纯 FSDP2 要求 BF16 逐位相同。EP 或 CP 使用固定的 `rtol=0.016, atol=0.03125`，同时输出位模式差异数与最大绝对误差。EP 和 CP 会改变每个 expert 的 GEMM 行数，默认 BLAS 设置在本次测试中超出了容差。分布式 bench 对本地模型和官方模型都使用 cuBLASLt，关闭 BF16 reduced-precision reduction 和 split-K。[PyTorch 2.11 的 BF16 GEMM 设置](https://docs.pytorch.org/docs/2.11/backends.html#torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)

小模型的 FP32 检查覆盖输入及参数梯度、零 token 的 expert、跨 rank 的 PLE 历史和 QSA 块筛选。Expert FSDP2 用 `set_gradient_divide_factor(world_size)` 将梯度归一化到各 rank 局部 mean loss 的平均值。`finish_gradients` 归一化 PLE 表的梯度，并在关闭 FSDP2 时归约参数副本。

```bash
PYTHONPATH=. torchrun --standalone --nproc-per-node=8 tests/my_chunked_ep/check_backward.py
```

## 实验记录

### ZJK 四层并行对拍

2026-11-23，单个 launcher 的 8 张 RTX PRO 5000 72GB，PyTorch 2.11.0+cu128、Transformers 5.24.0、Accelerate 1.16.0。从节点本地的 `/hf-hub/Qwen/Qwen3.8-Flash-Next` 读取前 4 层。每种模式新启一次 `torchrun`，运行前已读过 checkpoint。每个 CP 组输入为 `[2, 8]`，四组实验的最大绝对误差均为 0。

| 模式 | 各 rank 加载 (s) | 加载和 forward 最慢 (s) | 含官方对拍的命令全程 (s) | 每卡峰值 (GiB) | 本地 router 行数 / expert 数 | BF16 位差异 / logits 数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FSDP2 | 1.65–2.13 | 3.82 | 47.52 | 26.820 | 64 / 64 | 0 / 31,784,960 |
| EP=4 | 2.22–3.24 | 4.09 | 47.46 | 19.686 | 512 / 128 | 0 / 31,784,960 |
| CP=2 | 3.83–4.43 | 5.24 | 51.08 | 33.719 | 512 / 512 | 0 / 15,892,480 |
| FSDP2 + EP=4 + CP=2 | 1.69–2.42 | 4.11 | 50.89 | 19.789 | 64 / 64 | 0 / 15,892,480 |

每卡峰值取本地模型从加载到 forward 结束的 `max_memory_allocated`，同一模式的 8 个 rank 在表中精度下相同。组合模式每卡的 router 权重为 `[64, 2560]`，expert 的 `gate_up_proj` 为 `[64, 1280, 2560]`。四个模式都只常驻 PLE 表的 `[40000192, 160]` 行分片，每卡 11.92 GiB。

同一节点的单进程四层命令耗时 57.98 s。本地模型和官方模型分别加载 21.82 s 和 16.56 s，3,973,120 个 BF16 logits 全部逐位一致。

四层小模型在 8 卡以 FP32 通过 forward、输入梯度和参数梯度检查。CP=4，EP=4 时分别开关 FSDP2，EP=8 时验证单 rank expert FSDP mesh 的梯度除数。测试覆盖零发送和零接收的 EP rank、跨多个 rank 取 PLE 历史，以及 QSA 实际筛选的 token 与官方选择一致。PyTorch 2.11.0+cu128、NCCL 2.28.9 上，使用自定义除数的 multi-rank expert reduce-scatter 走 `SUM`，再由 FSDP2 归一化。原始分 rank 记录位于 `$MY_DFS/work/mylws/outputs/my_chunked_ep/20261123-parallel.KmIeGiy7/`。

### H20 单进程对拍

2026-11-23，单节点 8 卡 H20，PyTorch 2.11.0+cu128、Transformers 5.24.0、Accelerate 1.16.0。attention 和 MoE 使用 eager，causal convolution 和 gated delta rule 使用 PyTorch reference。

**单层调试**从启动 Python 到完成两次 forward 并退出耗时 **14.86 s**。本地模型加载 6.39 s，`pub_qwen` 加载 1.79 s，3,973,120 个 BF16 logits 全部逐位一致。

#### 完整 48 层

节点上的副本共 335.3 GiB，含 131 个 safetensors 分片。全部 144 个文件的 SHA-256 与 CephFS 源文件逐一匹配。

每组使用新的 Python 进程。启动前用 `fincore` 统计 safetensors 分片在 Linux 页缓存中的字节占比。本地磁盘的首次测量在新副本落盘并释放其文件页后开始。

| checkpoint 位置 | 启动时页缓存占比 | 本地模型加载 | `pub_qwen` 加载 | 两次加载合计 |
| --- | ---: | ---: | ---: | ---: |
| `/hf-hub` 首次 | 0.00% | 91.2 s | 54.8 s | 146.0 s |
| `/hf-hub` 再次 | 98.28% | 49.5 s | 42.8 s | 92.3 s |
| CephFS，复制前 | 98.30% | 48.9 s | 41.8 s | 90.7 s |
| CephFS，本地磁盘两组完成后 | 51.09% | 144.4 s | 58.5 s | 202.9 s |
| CephFS，再次运行 | 18.47% | 234.9 s | 60.2 s | 295.1 s |

复制前的 CephFS 基线在相同的 GPU 同步边界计时，两次加载的进程 `read_bytes` 增量均为 0。每种条件各测一次。五组输入均为 `[2, 8]`，输出均为 `[2, 8, 248320]`，每组的 3,973,120 个 BF16 logits 全部逐位一致。

原始日志位于 `$MY_DFS/work/mylws/outputs/my_chunked_ep/20261123-checkpoint-load.ZNsO0I8D/experiment.log`。
