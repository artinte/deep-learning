import numpy
import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import mnist


class Conv3x3:
    # A Convolution layer using 3x3 filters.
    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = numpy.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        """
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array
        """
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i : (i + 3), j : (j + 3)]
                yield im_region, i, j

    def forward(self, input):
        """
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        """
        self.last_input = input
        h, w = input.shape
        output = numpy.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = numpy.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backprop(self, dl_dout, learn_rate):
        """
        Performs a backward pass of the conv layer.
        - dl_dout is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        """
        dl_dfilters = numpy.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                dl_dfilters[f] += im_region * dl_dout[i, j, f]

        # Update filters.
        self.filters -= learn_rate * dl_dfilters

        # We aren't returning anything here since we use Conv3x3 as
        # the first layer in our CNN. Otherwise, we'd need to return
        # the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None


(x_train, y_train), (x_test, y_test) = mnist.load()

conv = Conv3x3(8)
output = conv.forward(x_train[0])
assert output.shape == (26, 26, 8)
