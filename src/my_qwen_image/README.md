# my_qwen_image

`my_qwen_image.transformer_qwenimage.QwenImageTransformer2DModel` 使用 Diffusers 0.43.0 的组件，可直接从本地 Qwen Image checkpoint 的 `transformer/` 目录加载权重。

See the [DiT test README](../../tests/my_qwen_image/README.md) for the Diffusers and SGLang comparisons.

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
