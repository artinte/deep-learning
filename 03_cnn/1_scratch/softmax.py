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
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input

        totals = numpy.dot(input, self.weights) + self.bias
        self.last_totals = totals

        exp = numpy.exp(totals)
        return exp / numpy.sum(exp, axis=0)

    def backprop(self, dl_dout, learn_rate):
        '''
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - dl_dout is the loss gradient for this layer's outputs.
        '''
        # We know only 1 element of dl_dout will be nonzero.
        for i, gradient in enumerate(dl_dout):
            if gradient == 0:
                continue

            t_exp = numpy.exp(self.last_totals)

            S = numpy.sum(t_exp)

            # Gradient of the softmax  function.
            dout_dt = -t_exp[i] * t_exp / (S * S)
            dout_dt[i] += t_exp[i] * (S - t_exp[i]) / (S * S)

        # Gradients of totals against weights and bias.
        dt_dw = self.last_input
        dt_db = 1
        dt_dinputs = self.weights
        # Graidents of the loss against totals.
        dl_dt = gradient * dout_dt
        # Gradients of loss against weights and bias.
        dl_dw = dt_dw[numpy.newaxis].T @ dl_dt[numpy.newaxis]
        dl_db = dl_dt * dt_db
        dl_dinputs = dt_dinputs @ dl_dt

        # Update weights and bias.
        self.weights -= learn_rate * dl_dw
        self.bias -= learn_rate * dl_db

        return dl_dinputs.reshape(self.last_input_shape)
