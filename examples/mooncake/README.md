# Mooncake quickstart

核心是分布式 KV 存储和内存数据传输：

- **Master**：管理存储节点和对象位置，不中转 Tensor。
- **Store 客户端**：贡献存储内存，按 key 执行 put/get/remove；内部使用 Transfer Engine。
- **Transfer Engine**：按远端地址读写内存，可独立使用，无需 Store Master。

两个示例均使用单机 GPU Tensor + RDMA，需要支持 GPUDirect RDMA 的环境。
固定使用 `cuda:0`，Transfer Engine 两个进程共享该 GPU；Store 存储池仍是 CPU 内存，关闭 memcpy 捷径。
使用 P2PHANDSHAKE 交换连接信息。在 Linux、已有 PyTorch 的环境，从仓库根目录执行：

```bash
python -m pip install -r examples/mooncake/requirements.txt

# 终端 1：保持运行，结束后 Ctrl-C。
bash examples/mooncake/master.sh

# 终端 2：按 key 存取 Tensor 的原始字节。
python examples/mooncake/store.py

# 独立示例：启动两个进程，交换地址后直接传 Tensor。
python examples/mooncake/transfer.py
```

`store.py` 的 shape/dtype 由调用方掌握；`transfer.py` 的 Pipe 只传地址和完成信号。

两个 Python 示例均输出 `PASS` 即成功。

## 验证结果

2026-11-13 在 Linux H20 节点通过，环境为 Python 3.12、PyTorch 2.11.0、Mooncake 0.3.14，
使用独立 `/opt/venvs/tq-test` 环境运行上述命令。

- Store：GPU Tensor → CPU 存储池 → GPU Tensor，RDMA put/get、内容校验及删除通过。
- Transfer Engine：两个进程共享 GPU 0，通过 RDMA 写入 GPU Tensor，接收端内容校验通过。
- 日志确认 RDMA transport、握手及 `device='cuda:0'`。仅验证单机同步传输，未测跨节点、异步或性能；部分网卡探测及内存注销告警未影响校验。

日志：`$MY_DFS/work/mylws/outputs/mooncake/2026-11-13/gpu-rdma.log`。

参考：[Store API](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/api-reference/python/mooncake-store.md)、[Transfer Engine](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/index.md)。
