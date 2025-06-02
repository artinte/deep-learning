import numpy
from matplotlib import pyplot

mu = 2
sigma = numpy.sqrt(10)

x = numpy.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
pdf = (1 / (sigma * numpy.sqrt(2 * numpy.pi))) * \
    numpy.exp(-((x - mu)**2) / (2 * sigma**2))
pyplot.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.06)
pyplot.plot(x, pdf)
pyplot.grid()
pyplot.show()
