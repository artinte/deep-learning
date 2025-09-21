import torch
from matplotlib import pyplot
from custom_schedule import CustomSchedule

# 1. Define your model
model = torch.nn.Linear(10, 1)  # Still a simple linear model

# 2. Define your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

# 3. Define your loss function
criterion = torch.nn.MSELoss()

# 4. Instantiate the custom learning rate schedule
d_model = 512
warmup_steps = 4000
scheduler = CustomSchedule(optimizer, d_model, warmup_steps)

# Instead of completely random data, let's create data with a clear linear relationship
batch_size = 64
input_dim = 10
output_dim = 1

# Define true weights and bias that the model should ideally learn
true_weights = torch.tensor(
    [[0.5], [-0.2], [1.0], [0.3], [-0.7], [0.1], [0.9], [-0.4], [0.6], [-0.1]],
    dtype=torch.float32,
)
true_bias = torch.tensor([[2.0]], dtype=torch.float32)

# Generate inputs and targets based on a linear relationship plus some small noise
sim_inputs = torch.randn(batch_size, input_dim) * 5.0  # Scale inputs a bit
sim_targets = (
    sim_inputs @ true_weights + true_bias + torch.randn(batch_size, output_dim) * 0.05
)  # Add small noise


num_training_steps = 10000
learning_rates = []
training_losses = []  # List to store training loss values

print(f"Testing CustomSchedule for d_model={d_model}, warmup_steps={warmup_steps}")

for step in range(num_training_steps):
    inputs = sim_inputs  # Use the learnable simulated inputs
    targets = sim_targets  # Use the learnable simulated targets

    # 1. Zero out the gradients
    optimizer.zero_grad()

    # 2. Forward pass: compute predicted outputs by passing inputs to the model
    outputs = model(inputs)

    # 3. Compute the loss
    loss = criterion(outputs, targets)

    # 4. Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # 5. Perform a single optimization step (update model parameters)
    optimizer.step()

    # 6. Update the learning rate (call AFTER optimizer.step())
    scheduler.step()

    # Record the current learning rate and loss
    current_lr = optimizer.param_groups[0]["lr"]
    learning_rates.append(current_lr)
    training_losses.append(loss.item())

    # Print information at regular intervals
    if (step + 1) % 100 == 0 or step == 0:  # Print every 100 steps, and at step 0
        print(
            f"Step {step + 1}: Learning Rate = {current_lr:.8f}, Loss = {loss.item():.8f}"
        )  # Print more decimal places for loss

fig, axes = pyplot.subplots(1, 2)

# Plot Learning Rate Schedule
axes[0].plot(range(1, num_training_steps + 1), learning_rates)
axes[0].set_ylabel("Learning Rate")
axes[0].grid(True)

# Plot Training Loss
axes[1].plot(range(1, num_training_steps + 1), training_losses, color="red")
axes[1].set_ylabel("Loss")
axes[1].grid(True)
axes[1].set_yscale("log")  # Often helpful to see small changes when loss is low

pyplot.tight_layout()
pyplot.show()

# Check learned weights vs true weights
print("\nLearned Weights:")
print(model.weight.data)
print("\nTrue Weights:")
print(true_weights.T)  # Transpose true_weights for comparison with model.weight.data

print("\nLearned Bias:")
print(model.bias.data)
print("\nTrue Bias:")
print(true_bias)
