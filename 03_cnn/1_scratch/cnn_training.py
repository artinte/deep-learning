
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


