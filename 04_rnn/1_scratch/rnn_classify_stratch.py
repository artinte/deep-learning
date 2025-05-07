import numpy
import random
import sys
import os
from matplotlib import pyplot

def softmax(xs):
    # Applies the softmax function to the input array.
    return numpy.exp(xs) / sum(numpy.exp(xs))

print(numpy.round(softmax(numpy.array([1, 2, 3, 4])), 4))

class SimpleRNN:
    # A vanilla recurrent neural network.
    def __init__(self, input_size, output_size, hidden_size=64):
        rng = numpy.random.default_rng(seed=0)
        self.Whh = rng.standard_normal((hidden_size, hidden_size)) / 1000
        self.Wxh = rng.standard_normal((hidden_size, input_size)) / 1000
        self.Why = rng.standard_normal((output_size, hidden_size)) / 1000
        self.bh = numpy.zeros((hidden_size, 1))
        self.by = numpy.zeros((output_size, 1))
