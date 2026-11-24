# mylws

Lightweight implementations and examples for personal learning.

## Lightweight Implementations

- [my_chunked_ep](src/my_chunked_ep/README.md) provides standalone Chunked EP schedulers for PyTorch all-to-all and DeepEP.
- [my_fsdp](src/README.md#my_fsdp) implements per-parameter full sharding and gradient reduce-scatter in PyTorch.
- [myfa](src/README.md#myfa): TileLang Flash Attention with fixed-length and variable-length implementations.
- [my_fp8_quant](src/README.md#my_fp8_quant): 128×128 block-wise FP8 quantization in TileLang.

## Examples

- [CUDA Graph with autograd](examples/cuda_graph_autograd/README.md) captures a single function inside an eager training loop.
- [CuTe C++ basics](examples/cute/README.md) covers Layout, Tensor, slicing, tiling, thread partition, layout algebra, CUDA SGEMM, and Hopper WGMMA with TMA.
- [Ray + NIXL communication](examples/ray_nixl/README.md): Transfer GPU tensors between two actors using `ray.put` / `ray.get`.
- [TQ as KV](examples/tq/README.md): Read and write tensors by sample key and field name.
- [Mooncake quickstart](examples/mooncake/README.md): Master setup, Store KV reads and writes, and memory transfers with Transfer Engine.
- [TensorDict quickstart](examples/tq/README.md#tensordict-quickstart): Batch indexing, field operations, and batch transformations.
- [Langfuse quickstart](examples/langfuse/README.md): Trace a recursive tree of questions and generated answers.
