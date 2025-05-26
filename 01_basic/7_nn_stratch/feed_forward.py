import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = numpy.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class OurNeuralNetwork:
    '''
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
      - w = [0, 1]
      - b = 0
    '''
    def __init__(self):
        weights = numpy.array([0, 1])
        bias = 0

        # The Neuron class here is from the previous section.
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        
        # The inputs for o1 are the outputs from h1 and h2.
        out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2]))
        return out_o1


network = OurNeuralNetwork()
x = numpy.array([2, 3])
# 0.7216
print(round(network.feedforward(x), 4))

def mse_loss(y_pred, y_true):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_pred - y_true)**2).mean()

y_pred = numpy.array([0, 0, 0, 0])
y_true = numpy.array([0, 1, 1, 0])
# 0.5
print(mse_loss(y_pred, y_true))
