import torch
from matplotlib import pyplot

def gaussian(x):
    return torch.exp(-x**2 / 2)

def boxcar(x):
    return torch.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x

def epanechikov(x):
    return torch.max(1 - torch.abs(x), torch.zeros_like(x))

fig, axes = pyplot.subplots(1, 4, sharey=True, figsize=(9, 5))

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = torch.arange(-2.5, 2.5, 0.1)

for kernel, name, ax in zip(kernels, names, axes):
    ax.plot(x.numpy(), kernel(x).numpy())
    ax.set_xlabel(name)

pyplot.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.12)
pyplot.show()
