from matplotlib import pyplot
import numpy
import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import decision_boundary

data = numpy.array([
    [133, 65, 0],
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

def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + numpy.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
    '''
    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
    '''
    def __init__(self):
        rng = numpy.random.default_rng(0)
        # weights
        self.w1 = rng.random()
        self.w2 = rng.random()
        self.w3 = rng.random()
        self.w4 = rng.random()
        self.w5 = rng.random()
        self.w6 = rng.random()
        # biases
        self.b1 = rng.random()
        self.b2 = rng.random()
        self.b3 = rng.random()
    
    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, y_trues):
        learn_rate = 0.1
        epochs = 1000
        loss_list = []
        for epoch in range(epochs):
            for x, y_true in zip(data, y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                
                # Calculate partial derivatives.
                d_l_d_ypred = -2 * (y_true - y_pred)
                # neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                # neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                # neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                
                # Update weights and biaes.
                # neuron h1
                self.w1 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                # neuron h2
                self.w3 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                # neuron o1
                self.w5 -= learn_rate * d_l_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_l_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_l_d_ypred * d_ypred_d_b3
                
            if epoch % 10 == 0:
                y_preds = numpy.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(y_trues, y_preds)
                loss_list.append(loss)
        
        return loss_list

x_data = numpy.column_stack((x, y))

network = OurNeuralNetwork()
history = network.train(x_data, labels)

pyplot.plot(range(1, len(history) + 1), history)
pyplot.grid(True)
pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
pyplot.show()

for i in range(0, 3):
    temp = numpy.array([x[i], y[i]])
    print((data[i][0], data[i][1]), 'is',
          'female' if network.feedforward(temp) < 0.5 else 'male')

def predict(model, batch_x):
    result = []
    for x in batch_x:
        result.append(model.feedforward(x))
    return numpy.array(result)

decision_boundary.plot_single(x_data, labels,
                              lambda batch_x: predict(network, batch_x))
