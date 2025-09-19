import torch


# Suppose we have 3 classes
num_classes = 3

# Predicted scores (logits), not probabilities
# Shape: (batch_size, num_classes)
y_pred = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])

# Ground truth labels (as class indices)
# Shape: (batch_size,)
y_true = torch.tensor([0, 1])

criterion = torch.nn.CrossEntropyLoss()

loss = criterion(y_pred, y_true)
print(f"CrossEntropyLoss: {loss.item():.4f}")
