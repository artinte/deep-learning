from common import mnist
import numpy
import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(pathlib.Path(__file__))


class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values.
        self.weights = numpy.random.randn(input_len, nodes) / input_len
        self.bias = numpy.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input

        totals = numpy.dot(input, self.weights) + self.bias
        self.last_totals = totals

        exp = numpy.exp(totals)
        return exp / numpy.sum(exp, axis=0)

    def backprop(self, dl_dout):
        pass


(x_train, y_train), (x_test, y_test) = mnist.load()

