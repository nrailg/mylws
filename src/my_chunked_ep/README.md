# Chunked EP

这里的调度器来自 G-Core `0faff03f4`。`chunked_deepep.py` 使用 DeepEP Buffer 调度分块 dispatch 和 combine，`chunked_ep.py` 提供 PyTorch all-to-all 调度以及两条路径共享的 helper。

`eager_ep.py` 从 G-Core 同一版本的 `Qwen4ExpEPExperts._forward_eager` 提取默认路径，保留按 slot 交换、回传后加权的顺序，expert 计算通过 callback 传入。uneven all-to-all 及其反向原样来自 G-Core 的 `gpatch_v4/models/qwen4_exp/a2a.py`。

Qwen3.8-Flash-Next 的 checkpoint 实验和模型对拍说明在[归档](../../archived/model/README.md)。

## 独立 EP 对比

在支持 CUDA P2P 的单节点上运行。依赖 PyTorch、Transformers 的 `_grouped_linear`、DeepEP v1 的 `deep_ep.Buffer` 以及 fused slots 使用的 Triton。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 \
  tests/my_chunked_ep/check_deepep_speed.py
```

eager EP 和 Chunked DeepEP 共享 BF16 输入、top-k 路由和两组 BF16 expert 权重。每个 expert 按 G-Core `Qwen4ExpEPExperts._local_experts` 做 grouped Linear，先用 `gate_up_proj` 投影到 gate 和 up，再将 `SiLU(gate) * up` 送入 `down_proj`。测试通过随机初始化的权重运行。[grouped SwiGLU 实现](grouped_experts.py)

Chunked DeepEP 默认开启 G-Core fused slots。加 `--no-fused-slots` 可以保留相同的 grouped Linear，单独切换到 PyTorch gather 和 `index_add_`。用 `--tokens-per-rank` 调整每个 rank 的 token 数，用 `--comm-sms` 调整 DeepEP 通信 SM 数。`moe_slots.py` 原样复制自 G-Core `ae5807a96`。[算子源码](moe_slots.py)

先核对 forward 输出和四个梯度（hidden、routing weights、gate_up_proj、down_proj），再计时 forward + backward。预热 4 次后，交替测量 15 次。每次取各 rank 同步后的最大墙钟时间，表中为中位数。

2026-11-23，单机 8 张 H20，PyTorch 2.11.0+cu128、Transformers 5.5.4、DeepEP 1.2.1、Triton 3.6.0。hidden size 2048，expert intermediate size 512，512 experts，top-k 10，2 个 chunk。

| Fused slots | 通信 SM | Tokens/rank | eager (ms) | Chunked DeepEP (ms) | 延迟降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 开启 | 24 | 4096 | 15.453 | 13.194 | 14.6% |
| 关闭 | 24 | 4096 | 15.475 | 17.030 | -10.0% |
| 开启 | 32 | 4096 | 15.459 | 13.055 | 15.6% |
| 开启 | 24 | 16384 | 55.617 | 38.154 | 31.4% |
| 开启 | 32 | 16384 | 55.678 | 36.989 | 33.6% |

五组输出和梯度对拍均通过，最大相对 L2 误差低于 0.29%。关闭 fused slots 的配置触发了性能断言，延迟比 eager 高 10.0%。固定 24 个通信 SM 并开启 fused slots 时，每 rank 从 4096 增至 16384 个 token，延迟降幅从 14.6% 增至 31.4%。原始样本在 `$MY_DFS/work/mylws/outputs/my_chunked_ep/20261123-grouped-linear.MP8FMd/`。
