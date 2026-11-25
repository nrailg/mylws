# Qwen Image DiT tests

Run from the repository root. Set `QWEN_IMAGE_TRANSFORMER` to the local
`transformer/` directory of the full Qwen-Image checkpoint.

## Diffusers numerical alignment

Use Diffusers 0.43.0. The standalone regression was verified with PyTorch
2.11.0+cu128, Transformers 5.16.1, and Accelerate 1.16.0.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_ENABLE_PARALLEL_LOADING=YES PYTHONPATH=src \
  python tests/my_qwen_image/compare_diffusers.py "$QWEN_IMAGE_TRANSFORMER"
```

All three BF16 cases must match official Diffusers bitwise. The test uses
Torch SDPA, seed 42, batch size 1, 4,096 image tokens, and 256 text tokens.
It covers timesteps 1.0, 0.5, and 0.01, with a mask retaining 231 text tokens
in the final case.

## SGLang DiT numerical alignment

The comparison loads the full Qwen-Image transformer checkpoint in Diffusers,
`my_qwen_image`, and native SGLang v0.5.23. The models run sequentially with
BF16 and Torch SDPA on one GPU. All three forwards use the inputs and timesteps
from the Diffusers-only comparison above, including the 231-token text mask.

Use a SGLang v0.5.23 environment with Diffusers 0.43.0 and Accelerate 1.16.0.
Use the SGLang checkout at commit `297e54fd105e974cdf401e5b4d20253a22c16f90`.
The source export includes a `REVISION` file. If using a Git checkout, write
its revision with `git -C "$SGLANG_DIR" rev-parse HEAD > "$SGLANG_DIR/REVISION"`
after setting `SGLANG_DIR`.

```bash
SGLANG_VENV=/path/to/sglang-venv
SGLANG_DIR=/path/to/pinned/sglang
CUDA_HOME=/path/to/cuda-13.0
source "$SGLANG_VENV/bin/activate"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_ENABLE_PARALLEL_LOADING=YES \
  CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" \
  LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="src:$SGLANG_DIR/python" \
  python tests/my_qwen_image/compare_diffusers.py "$QWEN_IMAGE_TRANSFORMER" --sglang
```

The check requires bitwise equality with Diffusers and cosine similarity above
0.995 with SGLang `lossless` for each case. The output also records relative L2,
RMSE, maximum absolute error, and bitwise equality for SGLang. A failed SGLang
case also runs with `exact` quality to compare both references.

On 2026-11-25, all three cases passed on one RTX PRO 5000 72 GB with
PyTorch 2.14.1+cu130, Diffusers 0.43.0, Transformers 5.19.0, and Accelerate
1.16.0. The local and Diffusers outputs were bitwise equal in the three-way
comparison and in a separate Diffusers-only run on PyTorch 2.11.0+cu128.

| Timestep | Text mask | Cosine vs SGLang | Relative L2 | RMSE | Max abs | Bitwise equal |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1.0 | None | 0.99994403 | 0.00917828 | 0.00949988 | 0.06640625 | No |
| 0.5 | None | 0.99992329 | 0.01097799 | 0.01500906 | 0.3125 | No |
| 0.01 | 231 of 256 | 0.99994987 | 0.01413246 | 0.12368632 | 0.75 | No |

Log from both runs at
`$MY_DFS/work/mylws-qwen-parity.fh0UjG/outputs/my_qwen_image/sglang-20261125/integration.log`.

## VeOmni benchmark

`bench.py` measures single-GPU forward latency and compares outputs with
VeOmni. See the [model README](../../src/my_qwen_image/README.md) for the
command and results.
