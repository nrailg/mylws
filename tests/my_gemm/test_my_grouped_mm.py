import pytest
import torch
import torch.nn.functional as F

from my_gemm import my_grouped_mm


def _assert_close(actual: torch.Tensor, reference: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.detach().cpu(), reference.detach().bfloat16(), rtol=1 / 128, atol=5e-4
    )


@pytest.mark.parametrize(
    ("tokens", "in_features", "out_features", "ends"),
    [
        (151, 64, 128, [5, 5, 136, 147]),
        (8, 2048, 1024, [2, 2, 6, 8]),
        (8, 512, 2048, [2, 2, 6, 8]),
        (0, 32, 64, [0, 0, 0, 0]),
    ],
    ids=["ragged-tiles", "narrow-projection", "wide-projection", "empty-input"],
)
def test_grouped_linear(tokens: int, in_features: int, out_features: int, ends: list[int]) -> None:
    """用逐 expert 的 FP32 Linear 核对 2D 输入、3D 权重的输出及两侧梯度。"""
    torch.manual_seed(123)
    x = torch.randn(tokens, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(
        (len(ends), out_features, in_features),
        device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    offs = torch.tensor(ends, device="cuda", dtype=torch.int32)
    y = my_grouped_mm(x, w.mT, offs=offs)
    dy = torch.randn_like(y)
    y.backward(dy)

    x_ref = x.detach().cpu().float().requires_grad_()
    w_ref = w.detach().cpu().float().requires_grad_()
    parts = []
    start = 0
    for expert, end in enumerate(ends):
        parts.append(F.linear(x_ref[start:end], w_ref[expert]))
        if start == end:
            assert torch.count_nonzero(w.grad[expert]) == 0
        start = end
    parts.append(x_ref.new_zeros((tokens - start, out_features)))
    y_ref = torch.cat(parts)
    y_ref.backward(dy.cpu().float())
    _assert_close(y, y_ref)
    _assert_close(x.grad, x_ref.grad)
    _assert_close(w.grad, w_ref.grad)
