# Ray + NIXL

Producer 和 Consumer 是共享一张 GPU 的两个 Ray Actor。Driver 转交引用，Consumer 通过 NIXL 获取 CUDA Tensor。

```python
# Producer 发布；引用包在字典里，避免 Ray 自动解引用。
return {"ref": ray.put(payload, _tensor_transport="nixl")}

# Consumer 获取。
payload = ray.get(envelope["ref"])
```

需要 Linux、CUDA PyTorch、2 CPU 和 1 GPU。从仓库根目录运行：

```bash
python -m pip install -r examples/ray_nixl/requirements.txt
python examples/ray_nixl/main.py --size-mib 64
```

`verified: true` 表示数据和同卡校验通过。已有 Ray 集群加 `--address auto`。

[8 GiB NIXL / TCP 对比测试与结果](../../tests/ray_nixl/test_nixl_vs_tcp.md)（同卡测试，未验证跨节点 RDMA）。
