# CUDA Graph 与 autograd

用 `torch.cuda.make_graphed_callables()` capture `fn(x, weight)` 的 forward 和 backward。训练循环中的 `x * 2`、loss 和 optimizer step 走 eager。调用 `loss.backward()` 时，autograd 会 replay `fn` 的 backward graph，再沿 eager 的 `x * 2` 回传到 `x.grad`。将 `weight` 显式传给 `fn`，梯度会写入 `weight.grad`。

每轮传入不同的 `x` 和更新后的 `weight`，与纯 eager 路径比较输出、输入梯度、参数梯度及更新后的参数。实际参数与 capture 样本保持相同的 shape、dtype、device、layout 和 `requires_grad`。该 API 支持一阶梯度。

在已安装 CUDA 版本 PyTorch 且有至少一张 NVIDIA GPU 的环境中，从仓库根目录运行：

```bash
python examples/cuda_graph_autograd/main.py
```

成功时会打印 3 轮 `output and gradients match eager`。

2026-11-24 在 Gemini 单张 NVIDIA H20 上实测通过，三轮输出、输入和参数梯度及更新后的参数均与 eager 一致。环境为 Python 3.12.13、PyTorch 2.11.0+cu128（`/opt/venvs/carrot`）。

PyTorch 在 `make_graphed_callables()` 的 backward capture 阶段报出 `AccumulateGrad` stream mismatch 警告，本次 capture 和三轮 replay 均成功。日志位于 `$MY_DFS/work/mylws/outputs/cuda_graph_autograd.FD2Fer/h20.log`。

参考 [PyTorch 的 partial-network capture 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#partial-network-capture)。
