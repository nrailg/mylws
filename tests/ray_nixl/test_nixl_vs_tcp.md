# NIXL 与 TCP：8 GiB GPU Tensor

## 目的与边界

比较同一张 GPU 上两个 Ray Actor 交接一个 8 GiB float32 Tensor 的耗时。
两个 Actor 各申请 0.5 GPU，放在同一个含 2 CPU、1 GPU 的 placement group bundle。

- NIXL：每轮新建 ObjectRef，`ray.put(..., _tensor_transport="nixl")` → `ray.get()`，用 `set_target_for_ref` 接收到预分配 GPU buffer。
- TCP：原生 `AF_INET/SOCK_STREAM` socket，GPU → pinned CPU → TCP `127.0.0.1` → pinned CPU → GPU。预分配 host buffer，通过 memoryview 发送和 `recv_into` 接收，不使用 pickle 或 Ray object store 搬运 payload。
- 8 GiB = 8,589,934,592 字节，2,147,483,648 个 float32 元素。两个进程总共预分配 16 GiB GPU buffer、16 GiB pinned host buffer；校验另需临时显存。
- 计时从 driver 发起传输至接收端 GPU 同步完成；包含 Ray 控制调用、NIXL 发布/注册和 TCP 两端 GPU↔CPU 拷贝，排除数据生成、预分配、TCP 建连与数据校验。两条路径均复用已分配的接收 buffer。
- 两轮预热不参与统计，五轮正式测量交替执行顺序，输出全部样本、中位数和有效 GiB/s。每轮发布新引用、改变填充值并清空接收 buffer，防止旧数据或缓存命中造成虚假结果。
- 全量逐元素校验，并断言两端 node ID 和 GPU ID 相同。实际接收的 NIXL Tensor 必须落在指定 buffer。

这是单机共享 GPU 与 TCP loopback 的应用交接对比；不代表跨节点 TCP、RDMA 带宽，也不比较 NIXL 的不同 UCX transport。NIXL 实际使用的底层链路需另行观察。

参考：[Ray RDT](https://docs.ray.io/en/master/core/direct-transport/index.html)、[PyTorch copy_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.copy_.html)。

## 执行

按项目测试 skill 确认同步、环境、GPU 占用，暂停 dguard 并清理本次测试集群。
设置当前会话确认的 `MY_DFS` 和 `MYLWS_VENV` 后执行：

```bash
# 先验证 socket 和 NIXL 的数据路径。
bash tests/ray_nixl/test_nixl_vs_tcp.sh --size-mib 64 --warmup 1 --repeats 1

# 正式 8 GiB 对比。
bash tests/ray_nixl/test_nixl_vs_tcp.sh --size-mib 8192 --warmup 2 --repeats 5
```

## 2026-11-13

状态：**PASS**。64 MiB 冒烟测试与 8 GiB 正式测试均退出码 0；正式测试两条路径各 2 轮预热、5 轮计时，所有轮次逐元素校验通过。两端在同一节点，Ray GPU ID 均为 `["0"]`。

| 路径 | 中位耗时 | 有效吞吐 |
| --- | ---: | ---: |
| NIXL / UCX，同卡 | 16.834 ms | 475.23 GiB/s |
| 原生 TCP loopback，含 GPU↔CPU 拷贝 | 1907.133 ms | 4.195 GiB/s |

本次中位耗时比 TCP / NIXL 为 **113.29**。NIXL 的数值是同卡数据交接的有效吞吐，不是网络带宽。

正式计时原始样本（秒）：

```text
NIXL: 0.014962032, 0.016834033, 0.016312672, 0.017253718, 0.032561695
TCP:  1.825734861, 1.866125597, 1.907132867, 1.924362493, 1.959352844
```

NIXL 两轮预热分别为 5.411875 s、0.038140 s；TCP 为 2.016381 s、1.989088 s。冷启动不包含在表中的中位数。

实际环境：单节点 NVIDIA H20，Python 3.12.13、Ray 2.58.0、NIXL 1.5.2、PyTorch 2.11.0+cu128。解释器为 `/opt/venvs/ray-nixl-test/bin/python`；镜像 tag 未记录，上游源码 commit 未记录，依赖以实际安装版本为准。

仓库基线 commit：`ea1d90f57b7250345350f6456559270b3aa4c98b`；基准代码在运行时尚未提交，实际文件 SHA-256：

```text
test_nixl_vs_tcp.py: 00d472d29a0a5cdb4fcaa498bab2434f733d08142664a44f63a295333fdb4fc9
test_nixl_vs_tcp.sh: 7b4395ed76f2b160a6155c8be13f972d99793ffe3dac025e8fad21e4e5237b56
```

证据位于 `$MY_DFS/work/mylws/outputs/ray_nixl/2026-11-13/`：

- `nixl-vs-tcp-64mib-smoke.log`：小数据路径检查。
- `nixl-vs-tcp-8gib.log`：正式命令、版本、源码 hash、全部样本和 NIXL/UCX 日志。

运行结束已清理测试 Ray 进程并恢复 dguard。跨节点 TCP、RDMA 及具体 UCX lane 未验证。

## 2026-11-13 恢复后复测

从同步副本恢复上述三个测试文件，Python 和 shell 文件的 SHA-256 与上次记录完全一致。环境版本仍为上述 H20、Python 3.12.13、Ray 2.58.0、NIXL 1.5.2、PyTorch 2.11.0+cu128，未更新依赖。

状态：**PASS**。原示例 `--size-mib 64 --timeout 90`、64 MiB 对比冒烟测试、8 GiB 正式对比全部通过，执行任务退出码为 0。正式测试两条路径各 2 轮预热、5 轮计时，每轮全量校验通过，两端共用同一节点的 GPU 0。

| 路径 | 中位耗时 | 有效吞吐 |
| --- | ---: | ---: |
| NIXL / UCX，同卡 | 18.401 ms | 434.76 GiB/s |
| 原生 TCP loopback，含 GPU↔CPU 拷贝 | 1925.569 ms | 4.155 GiB/s |

TCP / NIXL 中位耗时比为 **104.65**。正式样本（秒）：

```text
NIXL: 0.019885767, 0.018400841, 0.017256371, 0.018351017, 0.024860633
TCP:  1.882688632, 1.884436714, 1.973242255, 1.925569405, 1.946791395
```

NIXL 两轮预热为 5.304359 s、0.020112 s；TCP 为 2.208148 s、1.951114 s。结果仍仅代表同卡与 loopback 下的应用交接耗时。

完整命令、依赖版本、源码 hash、三项测试输出和清理记录保存在 `$MY_DFS/work/mylws/outputs/ray_nixl/2026-11-13/nixl-vs-tcp-restored-1615.log`。任务结束确认无活动 Ray 进程，并已恢复 dguard。
