import sys
import pathlib
import random
from matplotlib import pyplot

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import mnist


(x_train, y_train), (x_test, y_test) = mnist.load()

fig = pyplot.figure(figsize=(x_train[0].shape[0] / 10,
                             x_train[0].shape[1] / 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(x_train[0], cmap='gray_r', vmin=0, vmax=255)
ax.axis('off')
pyplot.show()

images_show = []
labels_show = []
row = 5
col = 3
random.seed(100)
for i in range(0, row * 2):
    r = random.randint(0, len(x_train))
    images_show.append(x_train[r])
    labels_show.append(str(r) + ' ' + str(y_train[r]))
for i in range(0, row):
    r = random.randint(0, len(x_test))
    images_show.append(x_test[r])
    labels_show.append(str(r) + ' ' + str(y_test[r]))
index = 1
for image, label in zip(images_show, labels_show):
    ax = pyplot.subplot(col, row, index)
    ax.set_xticks([])  # remove x-axis ticks
    ax.set_yticks([])
    pyplot.imshow(image, cmap='gray')
    pyplot.title(label)
    index += 1
pyplot.tight_layout()
pyplot.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.01)
pyplot.show()

