import torch
import numpy
from matplotlib import pyplot

point_sample = torch.tensor([0.5, 1])


rng = numpy.random.default_rng(seed=0)
random_numbers = rng.standard_normal(size=100)
x_input_array = numpy.linspace(0, 4, 100)
y_true_array = 3 * x_input_array + 4 + random_numbers

point_dataset = numpy.stack((x_input_array, y_true_array), axis=-1)
assert point_dataset.shape == (100, 2)

pyplot.scatter(x_input_array, y_true_array, s=5, c="blue")
pyplot.plot(x_input_array, 3 * x_input_array + 4, c="red", label="f(x) = 3x + 4")
pyplot.legend()
pyplot.grid(True)
pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
pyplot.show()
