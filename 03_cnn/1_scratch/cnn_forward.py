import numpy
import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(pathlib.Path(__file__))

from common import mnist
from conv import Conv3x3
from max_pool import MaxPool2
from softmax import Softmax

(x_train, y_train), (x_test, y_test) = mnist.load()

conv = Conv3x3(8)                   # 28x28x1 -> 26x26x8
pool = MaxPool2()                   # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10

def forward(input, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    out = conv.forward(input / 255 - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -numpy.log(out[label])
    acc = 1 if numpy.argmax(out) == label else 0
    
    return out, loss, acc

loss = 0
num_correct = 0

for i, (image, label) in enumerate(zip(x_test[:400], y_test[:400])):
    out, l, acc = forward(image, label)
    loss += l
    num_correct += acc
    
    if (i + 1) % 100 == 0:
        print('Step ' + str(i + 1) + ': loss = ' + str(round(loss / 100, 4)) +
              ', accuracy = ' + str(num_correct / 100))
        loss = 0
        num_correct = 0
