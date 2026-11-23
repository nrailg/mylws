# Qwen3.8-Flash-Next checkpoint 加载与 forward 对拍

对拍使用 G-Core 对应的 `Qwen/Qwen3.8-Flash-Next` 文本模型 `Qwen4ExpForCausalLM`，共 48 层，每层从 512 个专家中选取 10 个。checkpoint 的配置和权重索引与公开版本 `812d96e` 一致。[checkpoint 版本](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/812d96e10a0dd82b69d749e68693698cf6c36daa) · [Transformers v5.24.0 源码](https://github.com/huggingface/transformers/tree/v5.24.0/src/transformers/models/qwen4_exp)

## 运行

运行需要 CUDA、PyTorch、Transformers 5.24.0、Accelerate 和完整的 checkpoint。在单个 launcher 的仓库根目录运行，默认读取节点上的 `/hf-hub/Qwen/Qwen3.8-Flash-Next`。日常对拍保留前 4 层，覆盖第 1～3 层的 GatedDeltaNet 和第 4 层带 QSA indexer 的 indexed attention。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1
python src/my_chunked_ep/model/qwen4_exp/bench.py --debug-num-layers 4
```

`--debug-num-layers N` 保留前 N 层及输入、输出权重。第 2 层含约 95 GiB 的 PLE embedding，单层调试会跳过这张表。Transformers 的加载报告会列出被裁掉层的参数，bench 仍会检查所有保留层。省略开关即加载全部 48 层。要复现 CephFS 对照，将 `QWEN_CHECKPOINT` 设为该模型在 CephFS 上的目录。

```bash
python src/my_chunked_ep/model/qwen4_exp/bench.py "$QWEN_CHECKPOINT"
```

本地模型先加载 BF16 权重并自动推断 device map，`pub_qwen` 随后从同一目录加载，复用该 device map 和已读入的文件页。每次计时从调用 `load_checkpoint` 开始，到持有模型权重的 GPU 完成同步为止。两次 forward 使用相同的 token，逐位比较全部 logits。[bench.py](model/qwen4_exp/bench.py)

## 实验记录

2026-11-23，单节点 8 卡 H20，PyTorch 2.11.0+cu128、Transformers 5.24.0、Accelerate 1.16.0。attention 和 MoE 使用 eager，causal convolution 和 gated delta rule 使用 PyTorch reference。

**单层调试**从启动 Python 到完成两次 forward 并退出耗时 **14.86 s**。本地模型加载 6.39 s，`pub_qwen` 加载 1.79 s，3,973,120 个 BF16 logits 全部逐位一致。

### 完整 48 层

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
