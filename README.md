# mylws

Lightweight implementations and examples for personal learning.

## Lightweight Implementations

- [myfa](src/README.md#myfa): TileLang Flash Attention with fixed-length and variable-length implementations.
- [my_fp8_quant](src/README.md#my_fp8_quant): 128×128 block-wise FP8 quantization in TileLang.

## Examples

- [CuTe C++ basics](examples/cute/README.md): Layout, Tensor, slicing, tiling, thread partition, layout algebra, and a CUDA SGEMM tutorial.
- [Ray + NIXL communication](examples/ray_nixl/README.md): Transfer GPU tensors between two actors using `ray.put` / `ray.get`.
- [TQ as KV](examples/tq/README.md): Read and write tensors by sample key and field name.
- [Mooncake quickstart](examples/mooncake/README.md): Master setup, Store KV reads and writes, and memory transfers with Transfer Engine.
- [TensorDict quickstart](examples/tq/README.md#tensordict-quickstart): Batch indexing, field operations, and batch transformations.
- [Langfuse quickstart](examples/langfuse/README.md): Trace a recursive tree of questions and generated answers.
