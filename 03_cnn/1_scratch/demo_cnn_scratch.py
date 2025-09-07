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

softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10
conv = Conv3x3(8)                   # 28x28x1 -> 26x26x8
pool = MaxPool2()                   # 26x26x8 -> 13x13x8

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


def train(iamge, label, lr=0.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    out, loss, acc = forward(iamge, label)
    
    # Calculate initial gradient.
    gradient = numpy.zeros(10)
    gradient[label] = -1 / out[label]
    
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)
    
    return loss, acc


# Train the model.
loss = 0
num_correct = 0

for epoch in range(10):
    print('Epoch ' + str(epoch + 1))

    for i, (image, label) in enumerate(zip(x_train[:5000], y_train[:5000])):
        if (i + 1) % 500 == 0:
            print('Step ' + str(i + 1) + ': loss = ' + str(round(loss / 500, 4)) +
                ', accuracy = ' + str(num_correct / 500))
            loss = 0
            num_correct = 0
        
        l, acc = train(image, label)
        loss += l
        num_correct += acc

# Test the model.
loss = 0
num_correct = 0
for i, (image, label) in enumerate(zip(x_test[:400], y_test[:400])):
    _, l, acc = forward(image, label)
    loss += l
    num_correct += acc

print('Test loss:', round(loss / 400, 4))
print('Test accuracy:', num_correct / 400)
