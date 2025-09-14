import numpy
from matplotlib import pyplot
import torch


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


x = numpy.linspace(-10, 10, 400)
pyplot.plot(x, sigmoid(x), label="f(x) = 1 / (1 + e^-x)")
pyplot.plot(x, deriv_sigmoid(x), label="f'(x) = f(x) * (1 - f(x))")
pyplot.legend()
pyplot.grid(True)
pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
pyplot.show()

print("The derivative at x = 2 from scratch is", numpy.round(deriv_sigmoid(2), 4))


x_val = torch.tensor([2.0], requires_grad=True)
output = torch.sigmoid(x_val)
output.backward()
print("The derivative at x = 2 using PyTorch is", numpy.round(x_val.grad.item(), 4))
