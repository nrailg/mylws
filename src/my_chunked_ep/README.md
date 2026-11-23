# Chunked EP

这里的调度器来自 G-Core `0faff03f4`。`chunked_deepep.py` 使用 DeepEP Buffer 调度分块 dispatch 和 combine，`chunked_ep.py` 提供 PyTorch all-to-all 调度以及两条路径共享的 helper。

`eager_ep.py` 从 G-Core 同一版本的 `Qwen4ExpEPExperts._forward_eager` 提取默认路径，保留按 slot 交换、回传后加权的顺序，expert 计算通过 callback 传入。uneven all-to-all 及其反向原样来自 G-Core 的 `gpatch_v4/models/qwen4_exp/a2a.py`。

Qwen3.8-Flash-Next 的 checkpoint 实验和模型对拍说明在[归档](../../archived/model/README.md)。

## 独立 EP 对比

在支持 CUDA P2P 的单节点上使用 PyTorch 和提供普通 `deep_ep.Buffer` API 的 DeepEP v1。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 \
  tests/my_chunked_ep/check_deepep_speed.py

PYTHONPATH=src OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 \
  tests/my_chunked_ep/check_deepep_speed.py --fused-slots
```

两条命令默认都使用 24 个通信 SM。用 `--comm-sms 32` 调整为 32 个 SM，用 `--tokens-per-rank 16384` 调整为每 rank 16384 个 token。启用 `--fused-slots` 时需要 Triton。`moe_slots.py` 原样复制自 G-Core `ae5807a96`，用 `gather_slots` 取出各 expert 所需的行，再用 `weighted_reduce_slots` 合并加权和求和。[算子源码](moe_slots.py)

两条路径使用相同的 BF16 token、top-k 路由和逐 expert 标量乘法。先核对输出、hidden、routing weights 和 expert scale 梯度，再比较 forward + backward 耗时。预热后交替计时，每条路径采样 15 次。每个样本取最慢 rank 的 CUDA 同步墙钟时间，最后比较中位数。

2026-11-23 在单机 8 张 H20 上通过。PyTorch 2.11.0+cu128、DeepEP 1.2.1、Triton 3.6.0，hidden size 2048，512 experts，top-k 10，2 个 chunk。

| Slot fusion | 通信 SM | Tokens/rank | G-Core eager (ms) | Chunked DeepEP (ms) | 延迟降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 关闭 | 24 | 4096 | 7.871 | 7.323 | 7.0% |
| 开启 | 24 | 4096 | 7.923 | 4.511 | 43.1% |
| 开启 | 32 | 4096 | 7.926 | 4.291 | 45.9% |
| 开启 | 32 | 16384 | 29.348 | 12.987 | 55.7% |

四组的输出和梯度对拍均通过，最大相对 L2 误差低于 0.22%。原始样本见 `$MY_DFS/work/mylws/outputs/my_chunked_ep/20261123-ep-sweep.UKw3Jq/repo-reproduction-0030.log`。
