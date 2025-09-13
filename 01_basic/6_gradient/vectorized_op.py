import numpy

# forward pass
W = numpy.random.randn(5, 10)
X = numpy.random.randn(10, 3)

D = W.dot(X)

# Now suppose we had the gradient on D from above in the circuit.
dD = numpy.random.randn(*D.shape)  # .T gives the transpose of the matrix
dW = dD.dot(X.T)

assert dD.shape == D.shape, "dD should have the same shape as D"
assert dW.shape == W.shape, "dW should have the same shape as W"
