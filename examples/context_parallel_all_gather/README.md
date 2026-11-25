# AllGather context parallel forward

## 实验结论

2026-11-24 使用相同的 B=1、global S=7168、Hq=Hkv=24、D=128、
每组 8 heads、noncausal、BF16 shape，完成了两组单机 8 卡测试。

### RTX PRO 5000 + FA4

- 默认 NCCL 配置下，`full_gather` 为 **3.049–3.067 ms**，`async` 为
  **2.879–2.910 ms**；同配置内约 **1.05–1.06×**，耗时降低约 5%。
- 固定 `NCCL_MIN_CTAS=NCCL_MAX_CTAS=16`、`NCCL_CGA_CLUSTER_SIZE=1` 时，
  本次测得 `async` 略快，为 **2.747–2.765 ms**。它比默认 `full_gather`
  约快 **1.11×**，耗时降低约 10%。
- 上述约 10% 包含两部分：固定 16 CTA 的 `full_gather` 为
  **2.909–2.912 ms**，比默认 `full_gather` 低约 5%；`async` 再比同配置
  `full_gather` 低约 5%。相对同配置分组 `serial`，`async` 约快 1.16×。
- 只设置 `NCCL_MAX_CTAS=8/16/24` 时，trace 显示 NCCL 实际都使用 4 CTA，
  因此那组计时不能用于判断 CTA 数量的影响。

### H20 + FA3

- `NCCL_MAX_CTAS=16`、`NCCL_CGA_CLUSTER_SIZE=1`、`sm_margin=8` 时，
  `async` 为 **0.992–0.997 ms**。
- 默认 NCCL 配置的 `full_gather` 为 **1.070–1.075 ms**。因此上述 `async`
  约快 **1.08×**，耗时降低约 7%。
- `NCCL_MAX_CTAS=24` 时 `async` 反而为 1.181–1.186 ms。增加通信 CTA
  不一定更快，通信与 FA3 的资源占用需要一起实测。

两组结果只适用于各自的 GPU、FA backend 和单机互联。H20 使用 FA3，RTX PRO 5000
使用 FA4；不能把跨 GPU 的耗时差只归因于 SM 数量或互联。

## 对照方法

三条路径完成相同的 attention：

- `full_gather`：一次 Gather 完整 K/V 后调用 FA。
- `serial`：K/V 按 8 heads 分组，Gather 和 FA 串行。
- `async`：分组计算当前 K/V 时预取下一组；日志中名称为 `overlap`。

每次独立运行采集 9 个 sample，每个 sample 包含 10 次 warmup 和 100 次计时
forward。CUDA event 覆盖输入 pack、Gather、FA 和输出 cat。每个 sample 先取
跨 rank 最大值，再对 9 个 sample 取 median。CTA 配置分别按 8/16/24 和
24/16/8 的顺序运行两轮；下表区间是两次 median 的范围，不是置信区间。

## RTX PRO 5000 结果

环境为单机 8 张 NVIDIA RTX PRO 5000 72GB Blackwell（每张 110 SM，CC 12.0）、
Python 3.12、PyTorch 2.11.0+cu128 和 NCCL 2.28.9。官方 FA4 wheel 环境为
`flash-attn-4==4.0.0b40`、`nvidia-cutlass-dsl==4.8.1`、
`quack-kernels==0.6.5`、`apache-tvm-ffi==0.1.16` 和
`torch-c-dlpack-ext==0.1.5`。

下面固定 `NCCL_MIN_CTAS=NCCL_MAX_CTAS`、`NCCL_CGA_CLUSTER_SIZE=1`。
FA4 SM120 没有 `sm_margin` 参数，三条路径均使用 `sm_margin=0`。
trace 确认三组 Gather grid 分别为 8/16/24 CTA。

| 固定 CTA | `full_gather` (ms) | `serial` (ms) | `async` (ms) |
| ---: | ---: | ---: | ---: |
| 8 | 2.917–3.000 | 3.229–3.265 | 2.785–2.829 |
| 16 | 2.909–2.912 | 3.188–3.209 | **2.747–2.765** |
| 24 | 2.927–2.941 | 3.218–3.221 | 2.770–2.771 |

16 和 24 CTA 的结果很接近；这里只能说本次 16 CTA 略快，不能推出普遍最优值。
固定 CTA 的两轮共 6 个 case 全部通过输出检查。

默认配置的 profiler trace 中，`full_gather` 每个 forward 的 FA 和 Gather
分别约为 0.455 ms 和 2.566 ms；分组后分别约为 0.670 ms 和 2.626 ms，
`async` 约有 17% 的 Gather 时间与 FA 重叠。通信占主要时间，但分组也增加 FA
总时间，因此通信较慢不代表会得到同等比例的端到端加速。这里的时间来自 profiling；
性能结论以上面的独立 benchmark 为准。

8 卡 correctness 已通过。测试期间发生过两次进程 SIGSEGV：其中取得调用栈的一次
发生在 NCCL 初始化的 `init_process_group` / `eager_connect_single_device`，
位于 attention 之前；另一次未取得调用栈，底层根因未确定。后续同代码 correctness
再次通过，固定 CTA 的 6 个 case 也全部成功；这不表示底层 NCCL 问题已经修复。

## H20 结果

环境为单机 8 张 H20（每张 78 SM，NVLink）、Python 3.12、
PyTorch 2.11.0+cu128、NCCL 2.28.9，以及 FA3 commit
`b038b4974f26ba3bc8ab3c8b840c60b0432ce39b`。

固定 `NCCL_CGA_CLUSTER_SIZE=1`，分别设置 `NCCL_MAX_CTAS=8/16/24`。
`full_gather` 使用 `sm_margin=0`，`serial` 和 `async` 使用
`sm_margin=8`。trace 确认三组 Gather grid 分别为 8/16/24 CTA。

| `NCCL_MAX_CTAS` | `full_gather` (ms) | `serial` (ms) | `async` (ms) |
| ---: | ---: | ---: | ---: |
| 8 | 1.293 | 1.454–1.455 | 1.058–1.062 |
| 16 | 1.100–1.101 | 1.250–1.251 | **0.992–0.997** |
| 24 | 1.076–1.077 | 1.195–1.272 | 1.181–1.186 |

默认 NCCL 配置（`NCCL_MAX_CTAS` 和 `NCCL_CGA_CLUSTER_SIZE` 均不设置）的
`full_gather` 是 **1.070–1.075 ms**。`NCCL_MAX_CTAS=16` 的 `async`
比同配置 `serial` 低约 20%，但整体收益使用默认 `full_gather` 作为基线。
24 CTA 的 `serial` 在两次运行间波动明显，表中保留两次 median。

额外的 SM margin sweep 显示：`NCCL_MAX_CTAS=16` 时，margin=0/4/8/16 的
`async` 都在 0.992–0.996 ms，margin=24 增至 1.766 ms。`NCCL_MAX_CTAS`
限制 NCCL kernel 的 CTA 数，不表示“少用多少个 SM”；`sm_margin` 也不是物理 SM 隔离。

2 卡和 8 卡 correctness 均已通过；接入 FA4 backend 后，原 FA3 路径又通过了
2 卡 correctness 回归。

## 实现范围

输入布局为 BSHD，只考虑 BF16、D=128。各 rank 持有连续的 sequence shard；
Q 留在本地，K/V 按 head group 做 AllGather。`async` 使用两个 receive buffer：
计算当前 group 时预取下一组，`comm_stream.wait_stream()` 防止覆盖仍被 FA 使用的
buffer，ready event 保护消费者。每个 group 的完整 K/V 只调用一次 FA，
不需要合并 partial softmax。

causal 模式仍然 Gather 全部 K/V，只在 FA 计算时让 rank r 使用截止到本地 Q 末尾的
K/V prefix，以匹配 bottom-right causal mask。连续切分会让各 rank 的 causal 计算量
不均衡，因此计时取跨 rank 最大值。noncausal 模式各 rank 工作量相同，更适合单独观察
通信 overlap。

本文实现的是 AllGather。Ulysses 的 all2all 是 sequence/head shard 的转置，通信内容与
数据布局不同。两者要与 attention overlap，都需要独立的计算分块和正确的 stream 依赖；
仅设置 `async_op=True` 不能证明 GPU 上实际发生 overlap。本示例支持 MHA/GQA 和
causal/noncausal，不支持 varlen 和 backward。

程序默认使用 `--backend fa3`；FA4 需要显式设置 `--backend fa4`。FA3 调用
`flash_attn_interface.flash_attn_func(..., num_splits=1, sm_margin=...)`。
在 Hopper forward 路径中，`sm_margin` 减少 persistent scheduler 的 CTA 数量，
为 NCCL 留出执行资源。FA4 SM120 没有对应参数，只接受 `sm_margin=0`，不做无效的
reserve sweep。两个 backend 的 BSHD 输入、Gather 分组和 stream 依赖相同。

还可以用
[`NCCL_MIN_CTAS`、`NCCL_MAX_CTAS` 与 `NCCL_CGA_CLUSTER_SIZE`](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2287/user-guide/docs/env.html#nccl-max-ctas)
控制通信 kernel 的 CTA 配置，但设置的上限不等于实际 grid，需要用 trace 验证。

实现思路参考 zhuzilin 的
[更适合 flash attenion 体质的长上下文训练方案](https://zhuanlan.zhihu.com/p/718486708)
和 [async AllGather PR #50](https://github.com/zhuzilin/ring-flash-attention/pull/50)。

## 复现

### RTX PRO 5000 / FA4

使用隔离环境 `/opt/venvs/cp-fa4`。先运行 8 卡 correctness，再复现固定 16 CTA；
将两个 CTA 环境变量同时改为 8 或 24，可复现表中另两组结果。去掉三个 NCCL
环境变量可复现默认配置。

```bash
source /opt/venvs/cp-fa4/bin/activate
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/context_parallel_all_gather/main.py \
  --backend fa4 --sm-margins 0 --check-only

NCCL_MIN_CTAS=16 NCCL_MAX_CTAS=16 NCCL_CGA_CLUSTER_SIZE=1 \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/context_parallel_all_gather/main.py \
  --backend fa4 --batch-size 1 --seq-len 7168 \
  --q-heads 24 --kv-heads 24 --heads-per-group 8 \
  --sm-margins 0 --warmup 10 --iters 100 --repeats 9 \
  --trace-dir outputs/context_parallel_all_gather/traces-pro5000

python examples/context_parallel_all_gather/trace_overlap.py \
  outputs/context_parallel_all_gather/traces-pro5000
```

### H20 / FA3

需要 CUDA PyTorch，以及官方仓库 `hopper/` 中的
[FA3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)。本次从源码执行
`python setup.py build_ext --inplace`，并将生成的 extension 和 Python 模块加入
Python import path。编译时保留 BF16 D=128、MHA/GQA、causal/noncausal 的 Hopper
forward，裁掉 backward、varlen、split、paged/append KV、local、softcap、FP16/FP8、
SM80 和 D≠128 变体。

使用隔离环境 `/opt/venvs/cp-fa3`：

```bash
source /opt/venvs/cp-fa3/bin/activate
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/context_parallel_all_gather/main.py \
  --backend fa3 --check-only

NCCL_MAX_CTAS=16 NCCL_CGA_CLUSTER_SIZE=1 \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/context_parallel_all_gather/main.py \
  --backend fa3 --batch-size 1 --seq-len 7168 \
  --q-heads 24 --kv-heads 24 --heads-per-group 8 \
  --sm-margins 8 --warmup 10 --iters 100 --repeats 9 \
  --trace-dir outputs/context_parallel_all_gather/traces-h20

python examples/context_parallel_all_gather/trace_overlap.py \
  outputs/context_parallel_all_gather/traces-h20
```

`--seq-len` 是 global sequence length；添加 `--causal` 可检查 causal 路径。
correctness 使用 FP32 SDPA math，覆盖 B=2、MHA/GQA、causal/noncausal 和不同
group 大小；FA3 还覆盖 margin=0/4/8/16。容差为 `atol=0.002, rtol=0.02`；
同一分组下 `serial`/`async` 要求逐元素完全一致。benchmark shape 的各变体
还会与一次 `full_gather` 的输出比较。通过独立 correctness 后，可用
`--skip-check` 避免重复检查。

## 限制与证据

- RTX PRO 5000 是单机 8 卡，但不是 NVLink。NCCL 实际混合使用
  `SHM/direct/direct` 和 `NET/IB`，部分跨 NUMA 通道使用 GDRDMA；不能称为纯 RDMA。
- H20 与 RTX PRO 5000 使用不同 FA backend，不能直接用两张表判断硬件优劣。
- 同配置 `serial`/`async` 用于隔离 overlap 的效果；整体收益与各环境的默认
  `full_gather` 比较。CTA 配置、head group、FA tile 和 backend 会共同影响结果。
- profiler 与计时分开运行。trace 会受到 profiling、rank 等待和 CPU launch 间隔影响；
  性能结论使用独立 benchmark。
- RTX PRO 5000 的环境、计时、correctness、trace 摘要、拓扑和启动失败日志位于
  `outputs/context_parallel_all_gather/pro5000.A8Kkx4/`。
- H20 最新计时和 trace 位于 `outputs/context_parallel_all_gather/h20-cta.RzD2LI/`；
  历史 correctness 证据位于
  `outputs/context_parallel_all_gather/h20-fa3.jrIiBo/confirmed.log`。
