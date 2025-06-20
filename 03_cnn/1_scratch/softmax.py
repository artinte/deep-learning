import numpy

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
        input = input.flatten()
        input_len, nodes = self.weights.shape
        
        totals = numpy.dot(input, self.weights) + self.bias
        exp = numpy.exp(totals)
        return exp / numpy.sum(exp, axis=0)
