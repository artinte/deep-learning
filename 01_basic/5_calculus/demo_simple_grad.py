import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3 * x + 1

y.backward()
assert x.grad == 7
