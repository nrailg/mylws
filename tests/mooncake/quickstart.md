# Mooncake quickstart 测试

2026-11-13：通过。Linux H20 节点，Python 3.12、PyTorch 2.11.0、Mooncake 0.3.14；使用独立 `/opt/venvs/tq-test` 环境。

- 命令：[quickstart](../../examples/mooncake/README.md)。
- Store：GPU Tensor → CPU 存储池 → GPU Tensor，RDMA put/get、内容校验及删除通过。
- Transfer Engine：两个进程共享 GPU 0，通过 RDMA 写入 GPU Tensor，接收端内容校验通过。
- 日志确认 RDMA transport、握手及 `device='cuda:0'`。单机同步传输，未测跨节点、异步或性能；部分网卡探测及内存注销告警未影响校验。
- 日志：`$MY_DFS/work/mylws/outputs/mooncake/2026-11-13/gpu-rdma.log`。
