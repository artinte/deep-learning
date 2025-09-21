import torch
from matplotlib import pyplot


class SimpleAdam:
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# Test function: f(x) = (x - 3)^2
x = torch.tensor([10.0], requires_grad=True)

optimizer = SimpleAdam([x], lr=0.1)
losses = []
xs = []
for step in range(150):
    optimizer.zero_grad()
    loss = (x - 3) ** 2
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    xs.append(x.item())
    if (step + 1) % 10 == 0:
        # Print every 10 step
        print(f"Step {step+1:03d} | x = {x.item():.5f} | loss = {loss.item():.5f}")
print(f"Adam.Scratch Final x: {x.item():.5f}, Final Loss: {((x - 3)**2).item():.5f}")

pyplot.plot(losses, label="loss")
pyplot.plot(xs, label="x")
pyplot.legend()
pyplot.grid(True)
pyplot.show()

# Another implementation with torch.optim.Adam
x_torch_adam = torch.tensor([10.0], requires_grad=True)
optimizer_torch_adam = torch.optim.Adam([x_torch_adam], lr=0.1)
losses_torch_adam = []
xs_torch_adam = []

for step in range(150):
    optimizer_torch_adam.zero_grad() # Same zero_grad call
    loss = (x_torch_adam - 3) ** 2
    loss.backward() # Same backward call
    optimizer_torch_adam.step() # Same step call

    losses_torch_adam.append(loss.item())
    xs_torch_adam.append(x_torch_adam.item())
    if (step + 1) % 50 == 0: # Print less frequently for brevity
        print(f"Step {step+1:03d} | x = {x_torch_adam.item():.5f} | loss = {loss.item():.5f}")
print(f"Torch.Adam Final x: {x_torch_adam.item():.5f}, Final Loss: {((x_torch_adam - 3)**2).item():.5f}\n")
