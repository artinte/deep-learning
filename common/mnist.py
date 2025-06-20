import array
import numpy
import struct
import pathlib
import sys
import os
import zipfile

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common import download

mnist_url = 'https://github.com/artinte/tiny-datasets/raw/develop/mnist.zip'

def mnist_read(images_path, labels_path):
    labels = []
    with open(labels_path, 'rb') as file:
        magic, size = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, got {}'.format(magic))
        labels = array.array('B', file.read())

    with open(images_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, got {}'.format(magic))
        image_data = array.array('B', file.read())

    images = []
    for k in range(size):
        images.append([0] * rows * cols)
    for j in range(size):
        img = numpy.array(image_data[j * rows * cols:(j + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[j][:] = img

    return numpy.array(images), numpy.array(labels)

def load():
    url = mnist_url

    file_path = download.download(
        url, sha1_hash='1f2d5ca7b198c0b293014eb59e9064509d7315b1')

    extract_dir = os.path.dirname(file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    base_dir = os.path.join(extract_dir, 'mnist')

    x_train, y_train = mnist_read(
        os.path.join(base_dir, 'train-images-idx3-ubyte'),
        os.path.join(base_dir, 'train-labels-idx1-ubyte'))

    x_test, y_test = mnist_read(
        os.path.join(base_dir, 't10k-images-idx3-ubyte'),
        os.path.join(base_dir, 't10k-labels-idx1-ubyte'))

    return (x_train, y_train), (x_test, y_test)
