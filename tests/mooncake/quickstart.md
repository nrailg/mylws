# Mooncake quickstart 测试

2026-11-13：通过。Linux H20 节点，Python 3.12、PyTorch 2.11.0、Mooncake 0.3.14；使用独立 `/opt/venvs/tq-test` 环境。

- 命令：[quickstart](../../examples/mooncake/README.md)。
- Store：Tensor 原始字节 put/get、内容校验、强制删除后 key 不存在。
- Transfer Engine：两个进程通过 TCP 写入 CPU Tensor，接收端内容校验通过；日志确认 TCP-only。
- 范围：单机 CPU buffer；Store 同进程读写可走 memcpy，未测跨节点、GPU 传输或性能。
- 日志：`$MY_DFS/work/mylws/outputs/mooncake/2026-11-13/quickstart.log`。
