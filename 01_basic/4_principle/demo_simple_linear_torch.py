import torch

# Create linear model (y = wx + b)
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Generate training data (y = 3x + 2 with noise)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
y_true = 3 * x + 2 + torch.randn_like(x) * 0.5  # With small noise

# Training loop with only 10 epochs
print("Starting training...")
for epoch in range(10):
    # Reset gradients
    optimizer.zero_grad()
    
    # Forward pass: predict values
    y_pred = model(x)
    
    # Calculate loss
    loss = criterion(y_pred, y_true)
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    # Print progress for each epoch
    weight = model.weight.item()
    bias = model.bias.item()
    print(f"Epoch {epoch+1:2d}: Weight = {weight:.4f}, Bias = {bias:.4f}, Loss = {loss.item():.6f}")

# Final results
print("\nTraining complete!")
print(f"Learned weight: {model.weight.item():.4f} (target: 3.0)")
print(f"Learned bias: {model.bias.item():.4f} (target: 2.0)")
