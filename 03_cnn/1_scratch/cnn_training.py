from common import mnist
import numpy
import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(pathlib.Path(__file__))


(x_train, y_train), (x_test, y_test) = mnist.load()

