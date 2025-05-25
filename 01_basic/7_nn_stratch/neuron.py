import numpy


def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + numpy.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = numpy.matmul(self.weights, inputs) + self.bias
        return sigmoid(total)


# w1 = 0, w2 = 1
weights = numpy.array([0, 1])
# b = 4
bias = 4
# x1 = 2, x2 = 3
x = numpy.array([2, 3])

neuron = Neuron(weights, bias)
# 0.999
print('Neuron output is', round(neuron.feedforward(x), 3))
