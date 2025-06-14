import numpy
import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(pathlib.Path(__file__))

from common import mnist
from conv import Conv3x3

class MaxPool2:
    # A Max Pooling layer with a 2x2 filter.
    
    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        h, w, num_filters = input.shape
        output = numpy.zeros((h // 2, w // 2, num_filters))
        
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = numpy.amax(im_region, axis=(0, 1))
      
        return output

(x_train, y_train), (x_test, y_test) = mnist.load()

conv = Conv3x3(8)
pool = MaxPool2()

output = conv.forward(x_train[0])
output = pool.forward(output)
assert output.shape == (13, 13, 8)
