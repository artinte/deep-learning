from matplotlib import pyplot
import numpy

data = numpy.array([[133, 65, 0],
    [160, 72, 1],
    [150, 70, 1],
    [145, 66, 0],
    [152, 70, 1],
    [145, 65, 0],
    [150, 64, 0],
    [155, 66, 1],
    [140, 64, 0],
    [130, 62, 0],
    [150, 68, 1],
    [140, 68, 0],
    [135, 66, 0],
    [160, 68, 1],
    [155, 68, 1],
    [145, 64, 0],
    [162, 74, 1],
    [155, 72, 1],
    [158, 70, 1],
    [144, 72, 0]])

x = data[:, 0]
y = data[:, 1]
labels = data[:, 2]

colors = ['red' if label == 0 else 'blue' for label in labels]
pyplot.scatter(x, y, c=colors)
pyplot.xlabel('Weight (lb)')
pyplot.ylabel('Height (in)')
pyplot.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.12)
pyplot.grid(True)
pyplot.show()

average_weight = int(round(x.sum() / len(x)))
average_height = int(round(y.sum() / len(y)))
print('Average weight:', average_weight)
print('Average height:', average_height)

x = x - average_weight
y = y - average_height
print('Processed weight:', x)
print('Processed height:', y)
