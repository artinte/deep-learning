import math
import numpy

w = [2,-3,-3]
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot))

# backward pass through the neuron (backpropagation)
# # gradient on dot variable, using the sigmoid gradient derivation
ddot = (1 - f) * f
# backprop into x
dx = [w[0] * ddot, w[1] * ddot]
# backprop into w
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot]
print(numpy.round(dx, 4))
print(numpy.round(dw, 4))
