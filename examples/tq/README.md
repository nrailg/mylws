# TensorDict 与 TQ

## TensorDict quickstart

TensorDict 将一组共享 batch 维的 Tensor 组织成一个整体，适合表示训练数据的一批样本。
[tensor_dict_demo.py](tensor_dict_demo.py) 使用两个样本，在 CPU 上独立运行。

在 mylws 根目录执行（当前 `.venv` 已有依赖；新环境先安装 `torch tensordict`）：

```bash
.venv/bin/python examples/tq/tensor_dict_demo.py
```

1. **组织字段**：`batch_size=[2]` 表示两个样本；`input_ids` 的形状为 `[2, 3]`，`score` 为 `[2]`。所有字段的前导维必须匹配 batch 维，后面的特征维可以不同；嵌套字段用 `batch["metrics", "score"]` 访问。
2. **统一索引**：`batch[0]` 取第一个样本的所有字段，布尔 mask 同时筛选所有字段；`select(...)` 选择字段。普通切片共享底层数据，独立修改前用 `clone()`。
3. **整体操作**：`torch.cat` 沿已有 batch 维拼接，`torch.stack` 新增 batch 维；`reshape` 只改变 batch 维，`.to(device)` 统一迁移设备，`.apply(fn)` 对每个叶子 Tensor 应用函数。

实测：`cat` 得到 batch `[4]`，`stack` 得到 batch `[2, 2]`，其中 `input_ids` 形状为 `[2, 2, 3]`。
与下面 TQ 示例的关系：TensorDict 承载一批样本的字段，TQ 负责按 key 存取这些样本。
参考：[官方入门](https://docs.pytorch.org/tensordict/stable/getting_started.html)。

## TQ as KV

两个样本：写入 Tensor → 按字段读取 → 补充 score → 读取单个样本 → 删除。
使用 SimpleStorage（CPU 存储），`meta` 是样本元数据，`data` 是实际 TensorDict。

在已有 PyTorch 的环境、mylws 根目录执行；TransferQueue 源码位于同级目录：

```bash
python -m pip install -r ../TransferQueue/requirements.txt
PYTHONPATH="$PWD/../TransferQueue:${PYTHONPATH:-}" python examples/tq/tq_as_kv.py
```

2026-11-13 H20 服务器实测 **PASS**，退出码 0；本例仍使用 CPU Tensor，未验证 GPU 数据传输。
环境：`/opt/venvs/tq-test`，Ray 2.58.0、PyTorch 2.11.0+cu128、TQ 0.1.13.dev0。
日志：`$MY_DFS/work/mylws/outputs/tq/2026-11-13/tq-as-kv-h20.log`。
