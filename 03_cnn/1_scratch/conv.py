import numpy

class Conv3x3:
    # A Convolution layer using 3x3 filters.
    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = numpy.random.randn(num_filters, 3, 3) / 9
