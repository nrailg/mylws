# AllGather context parallel forward

只考虑 BF16、D=128 的 BSHD 输入。各 rank 持有连续的 sequence shard；Q 留在本地，
K/V 按 head group 做 AllGather。支持 MHA/GQA、causal/noncausal，不含 varlen 和 backward。

参考 zhuzilin 的[更适合 flash attenion 体质的长上下文训练方案](https://zhuanlan.zhihu.com/p/718486708)
与 [async AllGather PR #50](https://github.com/zhuzilin/ring-flash-attention/pull/50)：
计算当前 head group 时，预取下一组的全局 K/V。两个 receive buffer 交替使用，
`comm_stream.wait_stream()` 防止覆盖仍在被 FA3 读取的 buffer，ready event 保护消费者。
每个 group 的完整 KV 只调用一次 FA3，不需要合并 partial softmax。

本文采用的是 AllGather；Ulysses 的 all2all 是 sequence/head shard 的转置，通信内容与数据布局不同。
两者要与 attention overlap 都需要独立的计算分块和正确的 stream 依赖，`async_op=True`
本身不能证明 GPU 上实际 overlap。

causal 模式下，rank r 使用截止到本地 Q 末尾的 KV prefix，匹配 FA3 的 bottom-right causal mask。
连续切分会导致各 rank 的 causal 计算量不均衡；计时取所有 rank 的最大值。
noncausal 模式各 rank 工作量相同，更便于单独研究通信 overlap。

需要至少两张 Hopper GPU、CUDA PyTorch，以及官方仓库 `hopper/` 中的
[FA3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)。
调用 `flash_attn_interface.flash_attn_func(..., num_splits=1, sm_margin=...)`。
在这个 Hopper forward 路径中，`sm_margin` 减少 persistent scheduler 的 CTA 数量，
为 NCCL 留出执行资源；它不是物理 SM 隔离，最佳值需要实测。
还可以用 [`NCCL_MAX_CTAS` 与 `NCCL_CGA_CLUSTER_SIZE`](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2287/user-guide/docs/env.html#nccl-max-ctas)
限制通信 kernel 的资源占用，但这也可能降低独立通信的带宽。所有对照必须使用相同配置，
并额外保留默认 NCCL 配置的全量 Gather，避免把通信限速误认为整体优化。

已验证的环境是单机 8 张 H20（每张 78 SM，NVLink）、Python 3.12、
PyTorch 2.11.0+cu128、NCCL 2.28.9，以及 FA3 commit
`b038b4974f26ba3bc8ab3c8b840c60b0432ce39b`。在已有 CUDA PyTorch 的环境中，
按官方 `hopper/` 构建步骤编译该 commit；本次用 `python setup.py build_ext --inplace`
从源码编译，将生成的 extension 和 Python 模块放入 Python import path。裁剪掉 backward、varlen、
split、paged/append KV、local、softcap、FP16/FP8、SM80 和 D≠128 的编译变体，
保留 BF16 D=128、MHA/GQA、causal/noncausal 的 Hopper forward。
本次测试使用隔离环境 `/opt/venvs/cp-fa3`；下列命令需先激活安装了 FA3 的环境。

```bash
source /opt/venvs/cp-fa3/bin/activate
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/context_parallel_all_gather/main.py --check-only
NCCL_MAX_CTAS=8 NCCL_CGA_CLUSTER_SIZE=1 \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/context_parallel_all_gather/main.py \
  --batch-size 1 --seq-len 7168 --q-heads 24 --kv-heads 24 --heads-per-group 8 \
  --sm-margins 8 --iters 100 --repeats 9 \
  --trace-dir outputs/context_parallel_all_gather/traces
python examples/context_parallel_all_gather/trace_overlap.py outputs/context_parallel_all_gather/traces
```

`--seq-len` 是 global sequence length；添加 `--causal` 测量 causal 路径。
默认先用 FP32 SDPA math 验证 B=2、MHA/GQA、两种 mask、不同 group 大小和 SM margin。
容差为 `atol=0.002, rtol=0.02`；同一分组下 serial/async 要求逐元素完全一致。
同时验证 benchmark shape 的各变体输出与一次全量 Gather 一致。
已通过独立 correctness 检查后，可用 `--skip-check` 避免扫参数时重复小 shape 检查。
将 `--sm-margins 8` 改为 `--sm-margins 0 4 8 16` 可比较 reserve SM 的代价。

计时包含输入 pack、NCCL、FA3 和输出拼接；复用通信 buffer，预热后用 CUDA event 测量。
每组样本先取跨 rank 最大值，再取多个样本的 median，交替正序/倒序测量。
`full_gather` 是一次全量 KV AllGather 的基线；`serial` 和 `overlap` 使用完全相同的 head 分块。
同一 SM margin 下比较 serial/overlap，另与 margin=0 和 full_gather 比较，区分 SM 限流的代价。

profiler 与计时分开运行，只采集 CUDA activity，连续记录三个 forward。
CPU operator tracing 会放大 launch 间隔，显著改变短 kernel 的 overlap。
trace 脚本统计 AllGather 与 FA3 GPU kernel 区间的交集，不是 CPU launch 区间；
NCCL kernel 也可能在等待其他 rank，因此 overlap 百分比不能替代端到端加速。

2026-11-24 验证通过：2 卡与 8 卡的 FP32 SDPA correctness 检查均通过，
包括 B=2、MHA/GQA、causal/noncausal、1/2/4 KV heads per group，以及 margin=0/4/8/16。
上面命令的 noncausal MHA shape 经过独立进程复测，最新结果如下。
时间包含 pack、Gather、FA3、cat；每个样本运行 100 次，取 9 个跨 rank 最大值的 median。

| 路径 | SM margin | forward (ms) |
| --- | ---: | ---: |
| full_gather，同样限制 NCCL CTA | 0 | 1.292 |
| serial，8 heads per group | 8 | 1.456 |
| async，8 heads per group | 8 | 1.059 |
| full_gather，默认 NCCL 配置 | 0 | 1.073 |

async 相对同配置 serial **1.38×**，耗时减少 **27.3%**；相对同配置 full_gather 为 1.22×。
默认 NCCL 配置的 full_gather 已经接近 async，因此不认为这里证明了对默认全量 Gather 的明显加速。
该 shape 的 margin=0/4/8/16 均约 1.06–1.07 ms；这组结果证明了通信 overlap，
不能把收益单独归因于 reserve SM。

CUDA trace 中 serial 的交集为 0，async 在八个 rank 上有 40.3–53.2% 的 Gather kernel 时间
与 FA3 重叠。rank 0 的三个 forward 中，FA3 总时长基本不变（2.398 → 2.396 ms），
Gather 与 FA3 的交集为 1.040 ms，未与 FA3 重叠的 Gather 时间从 1.866 降到 0.917 ms。
trace 下的时间受 profiling 和 rank 到达时差影响，性能结论使用上面的独立计时。

另一个 B=4、S=8192、Hq=Hkv=32、4 heads per group 的默认 NCCL 对照显示了 contention：
margin=0 的 serial/async 为 6.795/6.839 ms；margin=4 为 6.785/6.527 ms；
margin=16 为 9.143/7.739 ms。reserve 太多 SM 会增加 FA3 的计算轮数，反而拖慢整个 forward。
默认 B=1、S=8192、Hq=32、Hkv=8 的 GQA case 也没有稳定的明显收益。
head group 大小、tile 数、NCCL CTA 数与 SM margin 需要共同测量，不能假设 async 一定更快。

日志、完整样本、编译记录与 trace 位于
`outputs/context_parallel_all_gather/h20-fa3.jrIiBo/`：`final.log` 是最新计时与 overlap 统计，
`confirmed.log` 包含 correctness 与 margin sweep，`mha-cuda-trace/` 是 GPU trace，
`overlap-timeline.png` 是 rank 0 的实际时间线。仅验证单机 NVLink；没有验证跨节点或 backward。
