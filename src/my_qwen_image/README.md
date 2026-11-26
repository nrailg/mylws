# my_qwen_image

`my_qwen_image.transformer_qwenimage.QwenImageTransformer2DModel` 使用 Diffusers 0.43.0 的组件，可直接从本地 Qwen Image checkpoint 的 `transformer/` 目录加载权重。

See the [DiT test README](../../tests/my_qwen_image/README.md) for the Diffusers and SGLang comparisons.

## Optimized inference results

2026-11-25: the optimized implementation achieved **12.7–15.0% higher throughput** than SGLang v0.5.23 (`297e54f`) across the six measured combinations. Both used the complete 20,430,401,088-parameter checkpoint on the same RTX PRO 5000 72 GB GPU, PyTorch 2.14.1+cu130, Diffusers 0.43.0, BF16, and PyTorch SDPA FlashAttention. Inputs were batch size 1, 4096 image tokens, and 256 text tokens, with seed 42. Each result is the median of three batches of ten timed calls after four warmup calls. Loading, compilation, and graph capture are excluded.

| Execution | Workload | Local (ms) | SGLang (ms) | Local throughput / SGLang |
| --- | --- | ---: | ---: | ---: |
| Eager | Single forward | 471.319 | 542.005 | 1.1500 |
| Eager | Serial CFG pair | 937.016 | 1063.217 | 1.1347 |
| Full CUDA graph | Single forward | 474.016 | 536.049 | 1.1309 |
| Full CUDA graph | Serial CFG pair | 934.675 | 1070.735 | 1.1456 |
| Regional compile | Single forward | 474.704 | 535.071 | 1.1272 |
| Regional compile | Serial CFG pair | 936.039 | 1063.255 | 1.1359 |

SGLang used its default `lossless` quality with all 60 packed text QKV sites enabled and its native fused QK norm/RoPE kernel. Its SM12.x gate disabled all 120 linear + GELU sites. The local implementation enabled linear + GELU fusion. Local peak allocated memory was 41.4–41.6 GiB, versus 38.2–38.4 GiB for SGLang.

Numerical validation against official Diffusers requires finite outputs, relative L2 error at most 2%, and local relative L2 error no greater than SGLang's for each case.

| Timestep | Text mask | Local relative L2 | SGLang relative L2 | Local max absolute error | SGLang max absolute error |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.0 | None | 0.6436% | 0.9184% | 0.052734 | 0.066406 |
| 0.5 | None | 0.5552% | 1.1021% | 0.125 | 0.3125 |
| 0.01 | 231 valid / 256 | 0.5663% | 1.4297% | 0.375 | 0.75 |

All six focused GPU tests passed, including modulation broadcasting, strided QK/RoPE, packed projections, and compiled inference after an in-place checkpoint update. An actual PEFT 0.22.0 merge/unload regression passed. CUDA graph replay, regional compile, and serial CFG checks matched eager outputs bitwise on changed inputs. Regional compilation created zero new graphs during timed batches. The default local path also matched official Diffusers bitwise in all three full-checkpoint cases.

Run from the project root in the verified container. `QWEN_IMAGE_TRANSFORMER` may point to the persistent checkpoint or the staged node-local copy used below.

```bash
source /opt/venvs/qwen-image-sglang-0523/bin/activate
export SGLANG_SRC=/tmp/sglang-v0.5.23
export CUDA_HOME=/tmp/mylws-qwen-cuda130/nvidia/cu13
export PYTHONPATH="$SGLANG_SRC/python:$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MAX_JOBS=8 OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1
QWEN_IMAGE_TRANSFORMER=/tmp/mylws-qwen-image.dxtonX
python -m pytest -q tests/my_qwen_image/test_kernels.py
python -u tests/my_qwen_image/benchmark_sglang.py "$QWEN_IMAGE_TRANSFORMER" \
  --sglang-revision-file "$SGLANG_SRC/REVISION" \
  --models sglang local --modes eager graph compile
```

The pinned source export contains its `REVISION` file. The isolated CUDA toolkit uses nvcc 13.0.88. Logs and environment details are under `$MY_DFS/work/mylws/outputs/my_qwen_image/20261125-sglang-v0523/`, with final measurements in `benchmark.log`. Node-local paths are specific to this container.

### Linear + GELU ablation

2026-11-25: disabling only local linear + GELU fusion brought throughput within **1.1–2.5% of SGLang**. The other optimizations remained enabled. The enabled control, disabled variant, and fresh SGLang baseline used the same full checkpoint, GPU, runtime, inputs, and timing procedure described above.

| Execution | Workload | Local fusion on (ms) | Local fusion off (ms) | SGLang (ms) | Fusion-off throughput / SGLang |
| --- | --- | ---: | ---: | ---: | ---: |
| Eager | Single forward | 473.393 | 530.962 | 536.924 | 1.0112 |
| Eager | Serial CFG pair | 927.796 | 1052.455 | 1065.944 | 1.0128 |
| Full CUDA graph | Single forward | 474.638 | 532.645 | 539.994 | 1.0138 |
| Full CUDA graph | Serial CFG pair | 937.581 | 1052.521 | 1079.303 | 1.0254 |

For eager single forward, fusion saved 57.569 ms, accounting for 90.6% of the 63.531 ms gap between the enabled local implementation and SGLang in this run. The full CUDA graph result showed a similar 58.007 ms saving. This isolates linear + GELU fusion as the main source of the measured advantage.

The switch smoke test observed zero fused calls when disabled and four when enabled in a two-block model. Full-model logs reported 0/120 and 120/120 enabled sites respectively; SGLang had zero eligible sites. With fusion off, relative L2 errors against Diffusers were 0.6577%, 0.5689%, and 0.5071% for the three existing cases, passing the same numerical criteria. Changed-input graph replay and local CFG branch checks matched eager bitwise with both switch settings.

Using the environment above, run the control followed by the ablation and SGLang baseline:

```bash
python -u tests/my_qwen_image/benchmark_sglang.py "$QWEN_IMAGE_TRANSFORMER" \
  --models local --modes eager graph
python -u tests/my_qwen_image/benchmark_sglang.py "$QWEN_IMAGE_TRANSFORMER" \
  --sglang-revision-file "$SGLANG_SRC/REVISION" \
  --models local sglang --modes eager graph --disable-linear-gelu
```

Logs are under `$MY_DFS/work/mylws/outputs/my_qwen_image/gelu-ablation/`: `switch-smoke.log`, `enabled.log`, and `disabled-and-sglang.log`. This ablation uses eager execution and raw CUDA graphs; regional compilation can independently fuse the separate operations.

### Limitations

Measurements cover unmasked single-GPU DiT forward and serial CFG pairs. The numerical checks also cover text padding. Fusion changes BF16 rounding. At pointwise `rtol=0.02, atol=0.02`, the local cases have 14, 86, and 14919 mismatches out of 262144 values, versus 309, 4771, and 55279 for SGLang. Packed text weights add about 3.16 GiB of memory. CUDA graphs require recapture after changing shapes or model weights.

## 单卡 forward 对比

在 `mylws` 根目录运行。测试使用 PyTorch 2.11.0+cu128、Diffusers 0.43.0、Transformers 5.16.1、Accelerate 1.16.0 和 VeOmni `86c5ec1d`。

```bash
VEOMNI_DIR=/path/to/VeOmni
QWEN_IMAGE_TRANSFORMER=/path/to/Qwen-Image/transformer
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH="src:$VEOMNI_DIR" \
  python tests/my_qwen_image/bench.py "$QWEN_IMAGE_TRANSFORMER"
```

2026-11-23，在一张 72 GB RTX PRO 5000 Blackwell 上，两种实现加载同一份 Qwen Image 20B BF16 权重，使用 batch size 1、4096 个图像 token、256 个文本 token 和 PyTorch SDPA。输入为固定的随机 latent 和文本 embedding。预热 5 次后，用 CUDA Event 计时 15 次 DiT forward。

| 实现 | 中位数（ms） | 最小–最大（ms） | 峰值分配显存（GiB） |
| --- | ---: | ---: | ---: |
| `my_qwen_image` | 439.573 | 438.451–440.453 | 38.434 |
| VeOmni（`SP=1`） | 443.659 | 442.508–445.428 | 38.441 |

两种实现的输出逐位一致。日志位于 `$MY_DFS/work/mylws/outputs/my_qwen_image/pro5000.e3O4Eg/bench.log`。

## Diffusers 官方实现对齐

```bash
source /opt/venvs/qwen-image-test/bin/activate
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_ENABLE_PARALLEL_LOADING=YES PYTHONPATH=src \
  python tests/my_qwen_image/compare_diffusers.py "$QWEN_IMAGE_TRANSFORMER"
```

测试直接使用 `diffusers.QwenImageTransformer2DModel` 作为 reference。两种实现顺序加载相同权重，使用 BF16、PyTorch SDPA、seed 42、batch size 1、4096 个图像 token 和 256 个文本 token。三个 case 分别为 timestep `1.0`、`0.5`、`0.01`；最后一个 case 用 mask 保留 231 个文本 token。输出记录 `max_abs`、RMSE 和逐位一致性，并要求逐位一致。

2026-11-25，在 RTX PRO 5000 72 GB 上，使用 PyTorch 2.11.0+cu128、Diffusers 0.43.0、Transformers 5.16.1、Accelerate 1.16.0 和完整 Qwen-Image 20B 权重，三个 case 全部逐位一致，`max_abs=0`、`RMSE=0`，进程退出码为 0。日志位于 `$MY_DFS/work/mylws/outputs/my_qwen_image/diffusers-20261125/compare.log`。

验证范围为 DiT forward；未覆盖 text encoder、scheduler、VAE 与完整生成图片的 pipeline。

## Inference optimizations

Call `model.enable_inference_optimizations()` after loading weights and completing `.to(device).eval()`. The fused kernels target BF16 CUDA inference with gradients and CUDA autocast disabled. The implementation covers six optimizations in [the model](transformer_qwenimage.py), [the inference kernels](inference_kernels.py), and [the benchmark](../../tests/my_qwen_image/benchmark_sglang.py):

1. **Norm, modulation, gate, and residual fusion.** Triton `norm_modulate` combines LayerNorm with scale/shift; `residual_norm_modulate` also folds in the preceding gated residual update; `gated_residual` combines the MLP gate and residual addition. These kernels reduce intermediate tensor traffic and kernel launches while retaining explicit BF16 rounding points for the elementwise operations.

2. **Direct writes into joint QKV buffers.** `project_joint_qkv` allocates Q/K/V buffers with `[text, image]` rows. For dense batch-size-1 inputs, the six text/image projections use `torch.addmm(..., out=...)` to write directly into their destination slices, removing the separate projection buffers and subsequent `cat`. The masked path keeps the joint layout, with packed text projection as described below; padding-mask semantics are preserved.

3. **QK RMSNorm + RoPE fusion.** `qk_norm_rope_` normalizes and rotates Q/K in place in the joint buffers, separately for text and image. It combines FP32 RMS reduction, learned scaling, and RoPE in one Triton launch per stream, preserving the intermediate BF16 casts. This path was verified on the RTX PRO 5000 SM12.0 GPU.

4. **Packed text QKV and linear + GELU.** When a text mask is supplied, eligible batch-size-1 inference uses cached concatenated text Q/K/V weights for one GEMM, then copies its three outputs into the joint buffers. The image and text MLPs use `torch._addmm_activation(..., use_gelu=True)` for the first linear layer and tanh GELU epilogue. The [full-model ablation](#linear--gelu-ablation) measures this fusion's contribution; `fuse_linear_gelu=False` disables it independently.

5. **Reuse RoPE and serial CFG conditioning.** `prepare_inference_rope` prepares position embeddings for reuse across forwards with fixed dimensions. `forward_cfg` shares the timestep embedding and each block's modulation between the positive and negative branches of one CFG pair. Modulation is recomputed for every pair, so a new timestep receives fresh conditioning.

6. **Regional `torch.compile` and full CUDA graph execution.** The benchmark provides block-level compilation and full-forward graph capture for both single forward and serial CFG. Compilation reduces Python dispatch through compiled regions; graph replay reuses a captured launch sequence and updates static input buffers. These modes are selected explicitly with `--modes`; their measured results are in the performance table above.

For a linear + GELU ablation, call `model.enable_inference_optimizations(fuse_linear_gelu=False)` before compiling or capturing CUDA graphs. The benchmark exposes the same setting as `--disable-linear-gelu`.

Prepare RoPE for fixed input dimensions before timing or CUDA graph capture.

```python
model.enable_inference_optimizations()
rope = model.prepare_inference_rope(img_shapes, text.shape[1])
with torch.inference_mode():
    output = model(
        hidden_states=latents,
        encoder_hidden_states=text,
        timestep=timestep,
        img_shapes=img_shapes,
        image_rotary_emb=rope,
        return_dict=False,
    )[0]
```

`model.forward_cfg(latents, text, negative_text, timestep, img_shapes=img_shapes, image_rotary_emb=rope)` returns the positive and negative predictions. It shares the timestep embedding and per-layer modulation within that pair. Each call computes modulation for its current timestep. Reusing RoPE requires matching image dimensions, text length, and device.

`tests/my_qwen_image/benchmark_sglang.py` supports eager, full CUDA graph, and regional `torch.compile` execution. CUDA graph replay updates input tensors while keeping shapes and model weights fixed. Recapture after changing weights. Regional compilation calls `compile(dynamic=False, options={"triton.cudagraphs": False})` on each transformer block.

## SGLang 优化参考

2026-11-25 核对 [SGLang v0.5.23](https://github.com/sgl-project/sglang/releases/tag/v0.5.23)，commit `297e54fd105e974cdf401e5b4d20253a22c16f90`。本地 checkout 位于 `~/work/sglang`。

| 优化 | 作用与适用条件 |
| --- | --- |
| LayerNorm、scale/shift、gate、residual fusion | 减少中间 tensor 和 kernel launch，覆盖 DiT block 的 modulation 路径 |
| QK RMSNorm + RoPE fusion | 合并 norm 与位置编码；具体 fast path 受硬件、dtype 和 layout 限制 |
| joint QKV buffer | batch size 1、无 mask、单卡 inference 等条件下直接写入 attention buffer，省去 concatenate |
| text QKV projection、linear + GELU fusion | 默认 `quality=lossless` 可启用，允许 rounding 顺序变化 |
| RoPE 与 modulation cache | 复用 RoPE；serial CFG 两次 forward 可复用同 timestep 的 modulation |
| TP、Ulysses/Ring SP、CFG parallel | 分配多卡计算与通信；需要独立测量多卡收益 |
| `torch.compile`、breakable CUDA graph | 降低执行开销；compile 默认关闭，graph 受 warmup、resident memory 等条件约束 |
| FP8/NVFP4、Nunchaku、Cache-DiT、progressive resolution | 量化或近似计算，需要另行验证质量 |

主要入口：[Qwen Image DiT](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py)、[fusion quality gating](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py)、[server args](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/multimodal_gen/runtime/server_args/server_args.py)。modulation kernel 的已有优化见 [PR #21318](https://github.com/sgl-project/sglang/pull/21318)。

RTX PRO 5000 属于 SM12.x，该 release 默认使用 Torch SDPA。BF16 QKV epilogue 的特殊路径在模型中限定为 SM90，不能直接归因到这张卡。`lossless` 保留数学表达式和 operand precision，但不保证逐位一致；`exact` 约束 SGLang 自身 reference chain，也不代表已与 Diffusers 对齐。
