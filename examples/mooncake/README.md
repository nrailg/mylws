# Mooncake quickstart

核心是分布式 KV 存储和内存数据传输：

- **Master**：管理存储节点和对象位置，不中转 Tensor。
- **Store 客户端**：贡献存储内存，按 key 执行 put/get/remove；内部使用 Transfer Engine。
- **Transfer Engine**：按远端地址读写内存，可独立使用，无需 Store Master。

本例固定单机 CPU Tensor + TCP，用 P2PHANDSHAKE 交换连接信息；不依赖 Ray/TQ，不验证 GPU/RDMA。
Linux、已有 PyTorch 的环境，在仓库根目录执行：

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

两个 Python 示例均输出 `PASS` 即成功。[测试记录](../../tests/mooncake/quickstart.md)。

参考：[Store API](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/api-reference/python/mooncake-store.md)、[Transfer Engine](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/index.md)。
