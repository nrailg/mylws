# Ray + NIXL 通信示例

两个 Ray Actor 传递一个装有 Tensor 的字典，使用与 slime 类似的接口：

```python
# Producer Actor 内：发布实际数据。
ref = ray.put(payload, _tensor_transport="nixl")

# Consumer Actor 内：获取实际数据。
payload = ray.get(ref)
```

两端都设置 `enable_tensor_transport=True` 和 `num_gpus=0.5`，共享同一张 GPU，在两个进程间传递 CUDA Tensor。
一个 placement group bundle 预留 `2 CPU + 1 GPU`，两个 Actor 都放入这个 bundle，确保使用同一节点上的同一张 GPU。
调度机制见 [Ray fractional GPU 文档](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)。

```text
Driver ── publish() ──→ Producer
Driver ←─ {ref} ────── Producer
Driver ── {ref} ─────→ Consumer
                       │ ray.get(ref)
Producer ═══ NIXL ═══→ Consumer
Driver ←─ 校验结果 ─── Consumer
```

Driver 只获取引用和小型校验结果。实际 Tensor 保留在生产端，由消费端通过 NIXL 获取。

## 环境与安装

在 Linux 上运行，使用 Python 3.10+。示例固定实测通过的 Ray 2.58.0、NIXL 1.5.2；RDT 仍是实验性接口。已有 Ray 2.58.0 时无需升级。

```bash
python3 -m venv .venv
source .venv/bin/activate
# 已有匹配驱动的 CUDA PyTorch 环境时，直接安装示例依赖。
python -m pip install -r examples/ray_nixl/requirements.txt
```

新环境先安装与机器驱动匹配的 [PyTorch CUDA 构建](https://pytorch.org/get-started/locally/)，再安装示例依赖。集群所有节点需要相同的 Python、Ray、NIXL、PyTorch 环境；UCX/NIXL 后端依赖也必须可用。

## 单机运行

从仓库根目录执行：

```bash
# 两个 Actor 共享 1 张 GPU，默认 16 MiB。
python examples/ray_nixl/main.py

python examples/ray_nixl/main.py --size-mib 64
```

不指定 `--address` 时，由 Ray 初始化本地运行环境。输出 `verified: true` 表示形状、dtype、设备、全部元素及两端 node ID/GPU ID 一致性均通过校验。

`fetch_seconds` 是消费端 `ray.get` 加设备同步的耗时，包含首次连接等开销，不包含数据生成和后续校验；它不是裸链路带宽基准。

已有 Ray 集群时加 `--address auto`，两个 Actor 仍在同一张 GPU 上。当前示例已移除 `--cross-node`。默认等待超时为 120 秒，可用 `--timeout` 调整。示例结束清理自己的 Actor、placement group 和 driver 连接，不停止已有集群。

## 需要理解的细节

- **引用包在字典中。** Ray 会自动解析顶层 ObjectRef 参数；`{"ref": ref}` 保留引用，让 Consumer 在方法内部显式调用 `ray.get`。Driver 对 `publish.remote()` 的第一次 `ray.get` 取回的只是这个字典。
- **Tensor 的生命周期。** Producer 必须存活至消费完成；示例保留引用且不修改已发布的 Tensor。RDT 发布不等于创建不可变数据快照，也不提供文件持久化。
- **NIXL 不等于已启用 RDMA。** 实际路径取决于 NIXL 后端、UCX/libfabric、网卡和配置；单机可能走共享内存或 CUDA IPC，跨机也可能走 TCP。输出的 `transport: nixl` 仅表示代码选择了 NIXL。
- **共享 GPU 不等于共享 Tensor。** 两个 Actor 是独立进程，仍通过 NIXL 传数据。`num_gpus=0.5` 是 Ray 的调度配额，不限制各进程的显存占用；生产端和接收端的分配都要能放进这张卡。
- **固定 GPU→GPU。** 不提供 CPU fallback 或自动设备选择；资源和传输结果用 `assert` 校验，请勿使用 `python -O`。

## 实测记录

8 GiB 的 NIXL / 原生 TCP socket 对比见 [测试说明与结果](../../tests/ray_nixl/test_nixl_vs_tcp.md)：

```bash
python tests/ray_nixl/test_nixl_vs_tcp.py --size-mib 8192 --warmup 2 --repeats 5
```

两个 Actor 仍共享一张 GPU，TCP 使用 loopback；两端 pinned CPU buffer 和 GPU buffer 均预分配，计时包含 TCP 的 GPU↔CPU 拷贝。该测试不代表跨节点 RDMA 与 TCP 的性能差异。

2026-11-12，简化前的版本在单节点 NVIDIA H20 服务器运行，Python 3.12、Ray 2.58.0、NIXL 1.5.2、PyTorch 2.11.0+cu128。以下是旧版命令和结果；当前版本已移除 `--device`。两个用例均退出码为 0，`verified: true`，形状、dtype、设备及全部元素校验通过，日志确认 NIXL 后端为 UCX。

| 用例 | 命令参数 | 结果 |
| --- | --- | --- |
| CPU→CPU，16 MiB | `--device cpu --size-mib 16` | 通过 |
| GPU→GPU，64 MiB，两张 GPU | `--device cuda --size-mib 64` | 通过 |

这是单机正确性测试，没有验证跨节点通信或 RDMA 性能。`fetch_seconds` 包含首次初始化开销，不用作带宽结论。

本次远端测试环境为 `/opt/venvs/ray-nixl-test`，通过 `.pth` 复用 `/opt/venvs/carrot` 中的 Ray/PyTorch，仅额外安装 NIXL；原训练环境未升级。复现时使用该环境的 `bin/python`。原始日志保存在检测得到的 `$MY_DFS/work/mylws/outputs/ray_nixl/2026-11-12/` 下，分别为 `cpu-16mib.log` 和 `cuda-64mib.log`。

2026-11-12，固定 GPU 路径后在同一环境回归 `python examples/ray_nixl/main.py --size-mib 64 --timeout 90`：**PASS**，退出码 0，全部断言通过，NIXL 后端为 UCX。源码 SHA-256 为 `14262257c70513597dc756d07bca7dea0071b48aa9ff88bd626e481446dc4028`；日志为同目录 `cuda-only-64mib.log`。仍未验证跨节点传输。

2026-11-13，共享 GPU 版本在同一 H20 环境运行 `python examples/ray_nixl/main.py --size-mib 64 --timeout 90`：**PASS**，退出码 0，全部数据校验通过，两端 node ID 相同，`source_gpu` 和 `destination_gpu` 均为 `["0"]`。日志确认 NIXL/UCX，两个 Actor 的 PID 分别为 875405、875406。源码 SHA-256 为 `2ef223d0a8ea0063b89c37a9b694775260cc3b279156fb4d9e48bf34a6ce5e2b`；日志保存在 `$MY_DFS/work/mylws/outputs/ray_nixl/2026-11-13/shared-gpu-64mib.log`。

参考：[Ray RDT 文档](https://docs.ray.io/en/master/core/direct-transport/index.html)、[Ray 2.58.0 NIXL 实现](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/experimental/rdt/nixl_tensor_transport.py)。
