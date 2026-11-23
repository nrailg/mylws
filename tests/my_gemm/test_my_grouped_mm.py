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
        (151, 64, 128, [0, 5, 5, 136, 147]),
        (8, 2048, 1024, [3, 3, 8]),
        (8, 512, 2048, [3, 3, 8]),
        (0, 32, 64, [0, 0]),
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


def test_grouped_weight_gradient() -> None:
    """用逐 expert 的 dY.T @ X 核对 dW，并验证这个 2D × 2D 调用的一阶梯度。"""
    torch.manual_seed(234)
    tokens, in_features, out_features = 128, 64, 32
    x = torch.randn(tokens, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(tokens, out_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ends = [5, 5, tokens]
    offs = torch.tensor(ends, device="cuda", dtype=torch.int32)
    dw = my_grouped_mm(dy.T, x, offs=offs)
    upstream = torch.randn_like(dw)
    dw.backward(upstream)

    x_ref = x.detach().cpu().float().requires_grad_()
    dy_ref = dy.detach().cpu().float().requires_grad_()
    parts = []
    start = 0
    for end in ends:
        parts.append(dy_ref[start:end].T @ x_ref[start:end])
        start = end
    dw_ref = torch.stack(parts)
    dw_ref.backward(upstream.cpu().float())
    _assert_close(dw, dw_ref)
    _assert_close(x.grad, x_ref.grad)
    _assert_close(dy.grad, dy_ref.grad)
    assert torch.count_nonzero(dw[1]) == 0
