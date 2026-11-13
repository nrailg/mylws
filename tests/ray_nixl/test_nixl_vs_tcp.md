# NIXL vs TCP：8 GiB

2026-11-13 **PASS**：64 MiB 冒烟和 8 GiB 对比均通过全量校验，两个 Actor 共用一张 H20。
环境：Python 3.12.13、Ray 2.58.0、NIXL 1.5.2、PyTorch 2.11.0+cu128。

已准备上述 GPU 环境后，在仓库根目录运行：

```bash
python tests/ray_nixl/test_nixl_vs_tcp.py --size-mib 8192 --warmup 2 --repeats 5
```

预热 2 轮、测量 5 轮，每轮使用新数据并全量校验；两端共预分配 16 GiB 显存和 16 GiB pinned 内存，校验另需临时显存。
计时包含 Ray 控制调用、NIXL 发布及 TCP 的 GPU↔CPU 拷贝，排除数据生成、缓冲区分配、建连和校验。

| 路径 | 中位耗时 | 有效吞吐 |
| --- | ---: | ---: |
| NIXL / UCX，同卡 | 18.40 ms | 434.76 GiB/s |
| TCP loopback，含 GPU↔CPU 拷贝 | 1925.57 ms | 4.155 GiB/s |

耗时比 **104.65 倍**。仅代表同卡与 loopback 的应用交接，不代表跨节点或 RDMA 性能。

日志：`$MY_DFS/work/mylws/outputs/ray_nixl/2026-11-13/nixl-vs-tcp-restored-1615.log`。
