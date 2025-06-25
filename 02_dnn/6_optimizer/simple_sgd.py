import torch

# training data
x = 1
y = 2
w = 0
leanring_rate = 0.1

y_pred = w * x
loss = (y_pred - y) ** 2
dw = 2 * (w * x - y) * x
w = w - leanring_rate * dw

print(f"w: {w}, loss: {loss}")

# implementation with torch
x = torch.tensor(1.0)
y = torch.tensor(2.0)
# weight parameter with gradient tracking
w_torch = torch.tensor(0.0, requires_grad=True)
# define optimizer
optimizer = torch.optim.SGD([w_torch], lr=0.1)

# forward pass
y_pred = w_torch * x
loss = (y_pred - y) ** 2

# backward pass
loss.backward()
optimizer.step()

assert torch.isclose(w_torch, torch.tensor(w), atol=1e-6)
