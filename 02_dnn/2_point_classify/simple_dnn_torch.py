import sklearn.datasets
import torch
import pathlib
import sys
import math
import numpy

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import decision_boundary

X_np, y_np = sklearn.datasets.make_moons(n_samples=200, noise=0.15)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.long)

num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2
epsilon = 0.1

class SimpleNeuralNetwork(torch.nn.Module):
    def __init__(self, nn_hdim):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(nn_input_dim, nn_hdim)
        self.fc2 = torch.nn.Linear(nn_hdim, nn_output_dim)
    
        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(0)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randn_like(
                self.fc1.weight) / math.sqrt(nn_input_dim))
            self.fc1.bias.zero_()

            self.fc2.weight.copy_(torch.randn_like(
                self.fc2.weight) / math.sqrt(self.fc2.in_features))
            self.fc2.bias.zero_()

    def forward(self, x):
        a1 = torch.tanh(self.fc1(x))
        out = self.fc2(a1)
        return out

def calculate_loss(model, X, y):
    criterion = torch.nn.CrossEntropyLoss()
    output = model(X)
    data_loss = criterion(output, y)
    return data_loss

def build_model(nn_hdim, num_passes=2000, print_loss=False):
    model = SimpleNeuralNetwork(nn_hdim)
    optimizer = torch.optim.SGD(model.parameters(), lr=epsilon)

    for i in range(num_passes):
        model.train()
        optimizer.zero_grad()
        loss = calculate_loss(model, X, y)
        loss.backward()
        optimizer.step()

        if print_loss and i % 100 == 0:
            print(f"Loss after iteration {i}: {loss.item():.4f}")
    return model

model = build_model(5, print_loss=True)

def predict(model, X):
    X = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        return numpy.argmax(logits.detach().numpy(), axis=1)

# Plot the decision boundary
decision_boundary.plot_single(X, y, lambda x: predict(model, x))
