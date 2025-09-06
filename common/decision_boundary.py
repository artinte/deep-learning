import numpy
from matplotlib import pyplot

# You will see that the background is divided into different color areas
# (indicating the model's predicted classification for different areas)
def internal_decision_plot(x, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                            numpy.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    pyplot.contourf(xx, yy, Z, cmap='Wistia', alpha=0.8)

def plot_single(x, y, pred_func):
    internal_decision_plot(x, pred_func)
    pyplot.grid(True)
    pyplot.scatter(x[:, 0], x[:, 1], c=y)
    pyplot.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.06)
    pyplot.show()

def plot_multi(x, y, pred_func):
    internal_decision_plot(x, pred_func)
    pyplot.axis('off')
    pyplot.scatter(x[:, 0], x[:, 1], c=y, s=10)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.02)
