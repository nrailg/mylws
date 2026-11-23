# Chunked EP

这里的调度器来自 G-Core `0faff03f4`。`chunked_deepep.py` 使用 DeepEP Buffer 调度分块 dispatch 和 combine，`chunked_ep.py` 提供 PyTorch all-to-all 调度以及两条路径共享的 helper。

Qwen3.8-Flash-Next 的 checkpoint 实验和模型对拍说明在[归档](../../archived/model/README.md)。
