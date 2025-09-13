import torch
from matplotlib import pyplot

data = torch.tensor(
    [
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
        [144, 72, 0],
    ],
    dtype=torch.float32,
)

x = data[:, 0]
y = data[:, 1]
labels = data[:, 2]

avg_weight = torch.round(x.mean()).item()
avg_height = torch.round(y.mean()).item()

x = x - avg_weight
y = y - avg_height
features = torch.stack([x, y], dim=1)
# Convert to a column vector.
targets = labels.view(-1, 1)

print("Average weight:", int(avg_weight))
print("Average height:", int(avg_height))


class OurNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sigmoid())
        self.output = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x


model = OurNeuralNetwork()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

history = []
epochs = 1000
for epoch in range(epochs):
    preds = model(features)
    loss = criterion(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        history.append(loss.item())

pyplot.plot(range(1, len(history) + 1), history)
pyplot.xlabel("Epoch (x10)")
pyplot.ylabel("MSE Loss")
pyplot.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.12)
pyplot.grid(True)
pyplot.show()
