import torch

# Predicted probabilities (output from a sigmoid layer)
y_pred = torch.tensor([0.9, 0.2, 0.1, 0.8], dtype=torch.float32)

# Ground truth labels (binary)
y_true = torch.tensor([1, 0, 0, 1], dtype=torch.float32)

# Define BCE loss
bce_loss = torch.nn.BCELoss()

# Compute loss
loss = bce_loss(y_pred, y_true)

# Print result
print("BCE Loss:", round(loss.item(), 4))
