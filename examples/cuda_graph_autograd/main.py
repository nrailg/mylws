import torch


def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x * weight).sin()


def main() -> None:
    assert torch.cuda.device_count() >= 1, "This example requires one CUDA GPU"
    device = torch.device("cuda:0")

    sample_x = torch.ones(8, device=device, requires_grad=True)
    sample_weight = torch.ones_like(sample_x, requires_grad=True)
    graphed_fn = torch.cuda.make_graphed_callables(fn, (sample_x, sample_weight))

    weight = torch.nn.Parameter(torch.full_like(sample_weight, 0.5))
    eager_weight = torch.nn.Parameter(weight.detach().clone())
    optimizer = torch.optim.SGD([weight], lr=0.1)
    eager_optimizer = torch.optim.SGD([eager_weight], lr=0.1)

    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        eager_optimizer.zero_grad(set_to_none=True)

        x = torch.linspace(0.1, 0.4, 8, device=device) * step
        x.requires_grad_()
        eager_x = x.detach().clone().requires_grad_()

        output = graphed_fn(x * 2, weight)
        eager_output = fn(eager_x * 2, eager_weight)
        loss = output.square().mean()
        eager_loss = eager_output.square().mean()
        loss.backward()
        eager_loss.backward()

        torch.testing.assert_close(output, eager_output)
        torch.testing.assert_close(x.grad, eager_x.grad)
        torch.testing.assert_close(weight.grad, eager_weight.grad)

        optimizer.step()
        eager_optimizer.step()
        torch.testing.assert_close(weight, eager_weight)
        print(f"step {step}: loss={loss.item():.6f}, output and gradients match eager")


if __name__ == "__main__":
    main()
