import numpy
import torch


def dervi_mse(y_pred, y_true):
    return 2 * (y_pred - y_true) / len(y_true)

y_true = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = numpy.array([1.2, 1.8, 3.5, 4.1, 5.3])

average_loss = ((y_pred - y_true) ** 2).mean()
print(f"Average loss from scratch: {average_loss:.4f}")

dl_dy = dervi_mse(y_pred, y_true)
print(dl_dy)

y_true = torch.tensor(y_true, requires_grad=False)
y_pred = torch.tensor(y_pred, requires_grad=True)

# update gradient
average_loss = torch.nn.MSELoss()(y_pred, y_true)
print(f"Average loss from torch: {average_loss.item():.4f}")

average_loss.backward()
print(y_pred.grad)
