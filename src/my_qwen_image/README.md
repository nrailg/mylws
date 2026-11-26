# my_qwen_image

`my_qwen_image.transformer_qwenimage.QwenImageTransformer2DModel` 使用 Diffusers 0.43.0 的组件，可直接从本地 Qwen Image checkpoint 的 `transformer/` 目录加载权重。

Diffusers 与 SGLang 的对比见 [DiT 测试说明](../../tests/my_qwen_image/README.md)。

## 推理优化结果

2026-11-25：优化后的实现在六组测试中，吞吐比 SGLang v0.5.23（`297e54f`）高 **12.7–15.0%**。两者均在同一张 RTX PRO 5000 72 GB GPU 上加载完整的 20,430,401,088 参数 checkpoint，使用 PyTorch 2.14.1+cu130、Diffusers 0.43.0、BF16 和 PyTorch SDPA FlashAttention。输入为 batch size 1、4096 个 image token、256 个 text token，seed 为 42。预热 4 次后，每组执行 3 轮、每轮计时 10 次，表中取 3 轮中位数；不计权重加载、编译和 graph capture。

| 执行方式 | 负载 | 本地实现（ms） | SGLang（ms） | 本地吞吐 / SGLang |
| --- | --- | ---: | ---: | ---: |
| Eager | 单次 forward | 471.319 | 542.005 | 1.1500 |
| Eager | 串行 CFG pair | 937.016 | 1063.217 | 1.1347 |
| 完整 CUDA graph | 单次 forward | 474.016 | 536.049 | 1.1309 |
| 完整 CUDA graph | 串行 CFG pair | 934.675 | 1070.735 | 1.1456 |
| Regional compile | 单次 forward | 474.704 | 535.071 | 1.1272 |
| Regional compile | 串行 CFG pair | 936.039 | 1063.255 | 1.1359 |

SGLang 使用默认的 `lossless` quality，启用了全部 60 个 packed text QKV site 和原生 fused QK norm/RoPE kernel；其 SM12.x 条件判断关闭了全部 120 个 linear + GELU site。本地实现启用了 linear + GELU fusion。本地实现峰值分配显存为 41.4–41.6 GiB，SGLang 为 38.2–38.4 GiB。

与官方 Diffusers 的数值验证要求输出均为有限值、relative L2 error 不超过 2%，并且每个 case 的本地 relative L2 error 不高于 SGLang。

| Timestep | Text mask | 本地 relative L2 | SGLang relative L2 | 本地最大绝对误差 | SGLang 最大绝对误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.0 | None | 0.6436% | 0.9184% | 0.052734 | 0.066406 |
| 0.5 | None | 0.5552% | 1.1021% | 0.125 | 0.3125 |
| 0.01 | 231 valid / 256 | 0.5663% | 1.4297% | 0.375 | 0.75 |

六项针对性 GPU 测试全部通过，覆盖 modulation broadcasting、strided QK/RoPE、packed projection，以及原地更新 checkpoint 后的 compiled inference。真实 PEFT 0.22.0 merge/unload 回归测试通过。对于变化后的输入，CUDA graph replay、regional compile 和串行 CFG 检查均与 eager 输出逐位一致。计时阶段 regional compile 没有创建新 graph。默认本地路径在三个完整 checkpoint case 中也都与官方 Diffusers 逐位一致。

在已验证的容器中从项目根目录运行。`QWEN_IMAGE_TRANSFORMER` 可以指向持久化 checkpoint，也可以指向下面使用的节点本地暂存副本。

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

固定版本的源码目录包含 `REVISION` 文件，隔离的 CUDA toolkit 使用 nvcc 13.0.88。日志和环境信息位于 `$MY_DFS/work/mylws/outputs/my_qwen_image/20261125-sglang-v0523/`，最终测量结果在 `benchmark.log` 中。节点本地路径仅适用于该容器。

### Linear + GELU 消融实验

2026-11-25：仅关闭本地 linear + GELU fusion 后，本地吞吐比 SGLang 高 **1.1–2.5%**。其余优化保持开启。开启 fusion 的对照组、关闭 fusion 的实验组和重新测量的 SGLang baseline 使用相同的完整 checkpoint、GPU、runtime、输入和上述计时方法。

| 执行方式 | 负载 | 本地开启 fusion（ms） | 本地关闭 fusion（ms） | SGLang（ms） | 关闭 fusion 后吞吐 / SGLang |
| --- | --- | ---: | ---: | ---: | ---: |
| Eager | 单次 forward | 473.393 | 530.962 | 536.924 | 1.0112 |
| Eager | 串行 CFG pair | 927.796 | 1052.455 | 1065.944 | 1.0128 |
| 完整 CUDA graph | 单次 forward | 474.638 | 532.645 | 539.994 | 1.0138 |
| 完整 CUDA graph | 串行 CFG pair | 937.581 | 1052.521 | 1079.303 | 1.0254 |

对于 eager 单次 forward，fusion 节省 57.569 ms，占本轮开启 fusion 的本地实现与 SGLang 之间 63.531 ms 差距的 90.6%。完整 CUDA graph 的结果相近，节省 58.007 ms。这说明 linear + GELU fusion 是本轮性能优势的主要来源。

开关 smoke test 使用两层模型：关闭时观测到 0 次 fused call，开启时观测到 4 次。完整模型日志分别记录 0/120 和 120/120 个启用 site；SGLang 的 eligible site 为 0。关闭 fusion 后，三个既有 case 相对 Diffusers 的 relative L2 error 分别为 0.6577%、0.5689% 和 0.5071%，通过相同的数值标准。两种开关状态下，变化输入的 graph replay 和本地 CFG 分支检查均与 eager 逐位一致。

使用上述环境，依次运行对照组、消融实验和 SGLang baseline：

```bash
python -u tests/my_qwen_image/benchmark_sglang.py "$QWEN_IMAGE_TRANSFORMER" \
  --models local --modes eager graph
python -u tests/my_qwen_image/benchmark_sglang.py "$QWEN_IMAGE_TRANSFORMER" \
  --sglang-revision-file "$SGLANG_SRC/REVISION" \
  --models local sglang --modes eager graph --disable-linear-gelu
```

日志位于 `$MY_DFS/work/mylws/outputs/my_qwen_image/gelu-ablation/`：`switch-smoke.log`、`enabled.log` 和 `disabled-and-sglang.log`。该消融实验使用 eager 和原始 CUDA graph；regional compile 可能独立融合这些分离的操作。

### 限制

性能测量覆盖无 mask 的单 GPU DiT forward 和串行 CFG pair，数值检查还覆盖 text padding。Fusion 会改变 BF16 舍入。在逐点 `rtol=0.02, atol=0.02` 标准下，本地三个 case 在 262144 个值中分别有 14、86 和 14919 个不匹配，SGLang 分别为 309、4771 和 55279 个。测量时使用的实现常驻了约 3.16 GiB packed text weights；当前实现仅在 masked input 出现时按需拼接这些权重，因此本文报告的耗时和显存数据来自此前的实现。CUDA graph 在 shape 或模型权重改变后需要重新 capture。

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

## 推理优化

加载权重并执行 `.to(device).eval()` 后，调用 `model.enable_inference_optimizations()`。Fused kernel 面向关闭梯度和 CUDA autocast 的 BF16 CUDA inference。实现分布在[模型](transformer_qwenimage.py)、[推理 kernel](inference_kernels.py)和 [benchmark](../../tests/my_qwen_image/benchmark_sglang.py)中，共包含六项优化：

1. **融合 norm、modulation、gate 和 residual。** Triton `norm_modulate` 合并 LayerNorm 与 scale/shift；`residual_norm_modulate` 进一步合并前一个 gated residual update；`gated_residual` 合并 MLP gate 与 residual addition。这些 kernel 减少中间 tensor 读写和 kernel launch，同时为逐元素运算保留显式 BF16 舍入点。

2. **直接写入 joint QKV buffer。** `project_joint_qkv` 分配行布局为 `[text, image]` 的 Q/K/V buffer。对于 dense batch-size-1 输入，六个 text/image projection 使用 `torch.addmm(..., out=...)` 直接写入目标切片，省去独立 projection buffer 和后续 `cat`。Masked path 保持相同 joint layout，并使用下述 packed text projection；padding mask 语义保持不变。

3. **融合 QK RMSNorm 与 RoPE。** `qk_norm_rope_` 分别对 text 和 image 的 joint buffer 切片原地完成 Q/K normalization 与 rotation。每条 stream 使用一次 Triton launch，合并 FP32 RMS reduction、learned scaling 和 RoPE，并保留中间 BF16 cast。该路径已在 RTX PRO 5000 SM12.0 GPU 上验证。

4. **Packed text QKV 与 linear + GELU。** 传入 text mask 时，符合条件的 batch-size-1 inference 会拼接当前 text Q/K/V 的 weight 和 bias，以一次 GEMM 计算三个 projection，再将输出复制到 joint buffer。Image 和 text MLP 的第一个 linear layer 使用 `torch._addmm_activation(..., use_gelu=True)` 执行 tanh GELU epilogue。[完整模型消融实验](#linear--gelu-消融实验)测量了该 fusion 的贡献；`fuse_linear_gelu=False` 可以单独关闭它。

5. **复用 RoPE 与串行 CFG conditioning。** `model.pos_embed` 为固定维度预先生成可跨 forward 复用的位置编码。`forward_cfg` 在一个 CFG pair 的 positive 与 negative 分支之间共享 timestep embedding 和每层 modulation。每个 CFG pair 都会重新计算 modulation，因此新 timestep 使用新的 conditioning。

6. **Regional `torch.compile` 与完整 CUDA graph。** Benchmark 为单次 forward 和串行 CFG 提供 block 级 compilation 与完整 forward graph capture。Compilation 通过 compiled region 减少 Python dispatch；graph replay 更新静态 input buffer，并复用已 capture 的 launch sequence。使用 `--modes` 显式选择这些模式；测量结果见上方性能表。

如需执行 linear + GELU 消融实验，请在 compile 或 capture CUDA graph 前调用 `model.enable_inference_optimizations(fuse_linear_gelu=False)`。Benchmark 通过 `--disable-linear-gelu` 暴露相同设置。

计时或 capture CUDA graph 前，为固定输入维度准备 RoPE。

```python
model.enable_inference_optimizations()
rope = model.pos_embed(img_shapes, max_txt_seq_len=text.shape[1], device=latents.device)
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

`model.forward_cfg(latents, text, negative_text, timestep, img_shapes=img_shapes, image_rotary_emb=rope)` 返回 positive 与 negative prediction，并在该 CFG pair 内共享 timestep embedding 和逐层 modulation。每次调用都会为当前 timestep 计算 modulation。复用 RoPE 时，image dimension、text length 和 device 必须一致。

`tests/my_qwen_image/benchmark_sglang.py` 支持 eager、完整 CUDA graph 和 regional `torch.compile`。CUDA graph replay 在 shape 和模型权重保持固定时更新输入 tensor；权重变化后需要重新 capture。Regional compilation 对每个 transformer block 调用 `compile(dynamic=False, options={"triton.cudagraphs": False})`。

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
