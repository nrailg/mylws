import pytest
import torch

from my_gemm import my_gemm


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(256, 384, 128), (136, 192, 80)],
    ids=["full-tiles", "tail-tiles"],
)
def test_my_gemm(m: int, n: int, k: int) -> None:
    """用 FP32 autograd 核对整块和尾块的 BF16 输出及两个输入的梯度。"""
    torch.manual_seed(123)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # 在 CPU 上计算 FP32 reference，避免 GPU 的 TF32 设置影响对照结果。
    a_ref = a.detach().to(device="cpu", dtype=torch.float32).requires_grad_()
    b_ref = b.detach().to(device="cpu", dtype=torch.float32).requires_grad_()
    c_ref = a_ref @ b_ref
    c = my_gemm(a, b)
    dc = torch.randn_like(c)
    c.backward(dc)
    c_ref.backward(dc.cpu().float())

    # 相对容差为一个 BF16 ULP 的上界，绝对容差覆盖接近零时的累加顺序差异。
    for actual, expected in ((c, c_ref), (a.grad, a_ref.grad), (b.grad, b_ref.grad)):
        torch.testing.assert_close(
            actual, expected.to(device=a.device, dtype=torch.bfloat16), rtol=1 / 128, atol=5e-4
        )


def test_unaligned_slice() -> None:
    """验证首地址未对齐的连续切片会触发断言。"""
    a = torch.empty(17, device="cuda", dtype=torch.bfloat16)[1:].view(2, 8)
    b = torch.empty(8, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="16-byte alignment"):
        my_gemm(a, b)


def test_unaligned_row_stride() -> None:
    """验证 BF16 行步长未按 16 字节对齐时会触发断言。"""
    a = torch.empty(8, 16, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(16, 17, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="16-byte row strides"):
        my_gemm(a, b)
