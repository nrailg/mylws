# Ray + NIXL 通信示例

两个 Ray Actor 传递一个装有 Tensor 的字典，使用与 slime 类似的接口：

```python
# Producer Actor 内：发布实际数据。
ref = ray.put(payload, _tensor_transport="nixl")

# Consumer Actor 内：获取实际数据。
payload = ray.get(ref)
```

两端都设置 `enable_tensor_transport=True`。示例默认传 CPU Tensor，也支持 CUDA Tensor。

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
python -m pip install -r examples/ray_nixl/requirements.txt
# CPU 示例：
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CUDA 模式改用与机器驱动匹配的 [PyTorch CUDA 构建](https://pytorch.org/get-started/locally/)，不要安装上面的 CPU 构建。集群所有节点需要相同的 Python、Ray、NIXL、PyTorch 环境；UCX/NIXL 后端依赖也必须可用。

## 单机运行

从仓库根目录执行：

```bash
# 2 个 CPU Actor，默认 16 MiB。
python examples/ray_nixl/main.py

# 每个 Actor 独占 1 张 GPU，共需 2 张 GPU。
python examples/ray_nixl/main.py --device cuda --size-mib 64
```

不指定 `--address` 时，由 Ray 初始化本地运行环境。输出 `verified: true` 表示形状、dtype、设备和全部元素都通过校验。结果包含实际字节数和两端 node ID。

`fetch_seconds` 是消费端 `ray.get` 加设备同步的耗时，包含首次连接等开销，不包含数据生成和后续校验；它不是裸链路带宽基准。

## 跨节点运行

在可信内网中启动 Ray 集群。以下占位符替换为节点实际 IP；两端需预留 CPU/GPU 资源。

```bash
# 节点 A
ray start --head --node-ip-address=<HEAD_IP> --port=6379

# 节点 B
ray start --address=<HEAD_IP>:6379 --node-ip-address=<WORKER_IP>

# 在节点 A、仓库根目录提交
python examples/ray_nixl/main.py --address auto --cross-node --size-mib 64

# 两个不同节点各需至少 1 张可用 GPU
python examples/ray_nixl/main.py --address auto --cross-node --device cuda --size-mib 64
```

`--cross-node` 使用硬性节点亲和性，将两个 Actor 放在两个不同节点，并检查返回的 node ID；不足两个合适节点时直接报错。默认等待超时为 120 秒，可用 `--timeout` 调整。示例结束只清理自己的 Actor 和 driver 连接，不停止已有集群。

## 需要理解的细节

- **引用包在字典中。** Ray 会自动解析顶层 ObjectRef 参数；`{"ref": ref}` 保留引用，让 Consumer 在方法内部显式调用 `ray.get`。Driver 对 `publish.remote()` 的第一次 `ray.get` 取回的只是这个字典。
- **Tensor 的生命周期。** Producer 必须存活至消费完成；示例保留引用且不修改已发布的 Tensor。RDT 发布不等于创建不可变数据快照，也不提供文件持久化。
- **NIXL 不等于已启用 RDMA。** 实际路径取决于 NIXL 后端、UCX/libfabric、网卡和配置；单机可能走共享内存或 CUDA IPC，跨机也可能走 TCP。输出的 `transport: nixl` 仅表示代码选择了 NIXL。
- **验证 RDMA 要看真实通路。** 使用 UCX 后端时，可在启动各节点 Ray 进程之前设置 `UCX_LOG_LEVEL=info`，结合 UCX 日志和网卡计数器检查；`--cross-node` 和传输成功本身都不是 RDMA 证据。不要将 `UCX_TLS` 随意限制为机器未提供的 transport。
- **CPU 和 GPU 是两条独立示例路径。** 默认 CPU→CPU 更接近分享对话里的 slime 数值字段；`--device cuda` 演示 GPU→GPU，不隐式执行 CPU→GPU 转换。

## 实测记录

2026-11-12，在单节点 NVIDIA H20 服务器运行，Python 3.12、Ray 2.58.0、NIXL 1.5.2、PyTorch 2.11.0+cu128。两个用例均退出码为 0，`verified: true`，形状、dtype、设备及全部元素校验通过，日志确认 NIXL 后端为 UCX。

| 用例 | 命令参数 | 结果 |
| --- | --- | --- |
| CPU→CPU，16 MiB | `--device cpu --size-mib 16` | 通过 |
| GPU→GPU，64 MiB，两张 GPU | `--device cuda --size-mib 64` | 通过 |

这是单机正确性测试，没有验证跨节点通信或 RDMA 性能。`fetch_seconds` 包含首次初始化开销，不用作带宽结论。

本次远端测试环境为 `/opt/venvs/ray-nixl-test`，通过 `.pth` 复用 `/opt/venvs/carrot` 中的 Ray/PyTorch，仅额外安装 NIXL；原训练环境未升级。复现时使用该环境的 `bin/python`。原始日志保存在检测得到的 `$MY_DFS/work/mylws/outputs/ray_nixl/2026-11-12/` 下，分别为 `cpu-16mib.log` 和 `cuda-64mib.log`。

参考：[Ray RDT 文档](https://docs.ray.io/en/master/core/direct-transport/index.html)、[Ray 2.58.0 NIXL 实现](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/experimental/rdt/nixl_tensor_transport.py)。
