import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# 1. Define a 1D Loss Function (as a PyTorch Module)
# This model has a single parameter 'x'. Its forward method returns our custom loss.
class LossLandscapeModel(torch.nn.Module):
    def __init__(self, initial_x):
        super(LossLandscapeModel, self).__init__()
        # 'x' is the parameter we want to optimize
        self.x = torch.nn.Parameter(torch.tensor(initial_x, dtype=torch.float32))

    def forward(self):
        x_val = self.x

        # Global minimum part: a parabolic well at x=2.0
        global_min_part = 0.5 * (x_val - 2.0) ** 2

        # Local minimum part: a small "dip" around x=0.0
        # We use a negative Gaussian to create this dip, making it a local minimum
        local_min_center = 0.0
        local_min_depth = -0.8  # Depth, negative value for a dip
        local_min_width = 0.3  # Width
        local_min_part = local_min_depth * torch.exp(
            -((x_val - local_min_center) ** 2) / (2 * local_min_width**2)
        )

        # A "hill" or "plateau" between the local and global minima
        # This makes it harder for SGD without momentum to pass through
        hill_center = 1.0
        hill_height = 1.0
        hill_width = 0.5
        hill_part = hill_height * torch.exp(
            -((x_val - hill_center) ** 2) / (2 * hill_width**2)
        )

        return global_min_part + local_min_part + hill_part


# 2. Set Optimization Parameters
initial_x_start = -1.5  # Starting point, ensuring it faces the local minimum first
learning_rate = 0.05
momentum_coeff = 0.9  # Momentum coefficient, used when momentum is enabled
num_iterations = 150  # Number of optimization steps for clearer convergence

# 3. Initialize Model and Optimizers
# SGD with Momentum
model_momentum = LossLandscapeModel(initial_x_start)
optimizer_momentum = optim.SGD(
    model_momentum.parameters(), lr=learning_rate, momentum=momentum_coeff
)

# SGD without Momentum
model_nomomentum = LossLandscapeModel(initial_x_start)
optimizer_nomomentum = optim.SGD(
    model_nomomentum.parameters(), lr=learning_rate
)  # momentum=0 or not specified means no momentum

# 4. Store History for Plotting
history_momentum_x = [model_momentum.x.item()]
history_momentum_loss = [model_momentum().item()]

history_nomomentum_x = [model_nomomentum.x.item()]
history_nomomentum_loss = [model_nomomentum().item()]

# 5. Training Loop (Optimization Steps)
print(f"Initial X: {initial_x_start}")
print(f"Learning Rate: {learning_rate}")
print(f"Momentum Coefficient: {momentum_coeff}")
print(f"Number of Iterations: {num_iterations}\n")

for i in range(num_iterations):
    # Optimization step for SGD with Momentum
    optimizer_momentum.zero_grad()  # Clear previous gradients
    loss_m = model_momentum()  # Compute loss
    loss_m.backward()  # Backpropagate to compute gradients
    optimizer_momentum.step()  # Update parameters
    history_momentum_x.append(model_momentum.x.item())
    history_momentum_loss.append(loss_m.item())

    # Optimization step for SGD without Momentum
    optimizer_nomomentum.zero_grad()
    loss_nm = model_nomomentum()
    loss_nm.backward()
    optimizer_nomomentum.step()
    history_nomomentum_x.append(model_nomomentum.x.item())
    history_nomomentum_loss.append(loss_nm.item())

# 6. Print Final Results
print(f'SGD with Momentum final x: {model_momentum.x.item():.4f}, final loss: {model_momentum().item():.4f}')
print(f'SGD without Momentum final x: {model_nomomentum.x.item():.4f}, final loss: {model_nomomentum().item():.4f}')

# 7. Visualization
# Numpy version of the loss function for plotting the curve
def plot_loss_function_np(x_val):
    global_min_part = 0.5 * (x_val - 2.0) ** 2
    local_min_center = 0.0
    local_min_depth = -0.8
    local_min_width = 0.3
    local_min_part = local_min_depth * np.exp(
        -((x_val - local_min_center) ** 2) / (2 * local_min_width**2)
    )
    hill_center = 1.0
    hill_height = 1.0
    hill_width = 0.5
    hill_part = hill_height * np.exp(
        -((x_val - hill_center) ** 2) / (2 * hill_width**2)
    )
    return global_min_part + local_min_part + hill_part


x_values = np.linspace(-3, 3, 500)
y_values = plot_loss_function_np(x_values)

plt.figure(figsize=(12, 7))
plt.plot(x_values, y_values, label="Loss Function", color="gray", linestyle="--")
plt.scatter(
    [2.0],
    [plot_loss_function_np(2.0)],
    color="red",
    marker="*",
    s=200,
    label="Global Minimum",
)
plt.scatter(
    [0.0],
    [plot_loss_function_np(0.0)],
    color="purple",
    marker="o",
    s=100,
    label="Local Minimum",
)

plt.plot(
    history_momentum_x,
    history_momentum_loss,
    color="blue",
    marker="o",
    linestyle="-",
    markersize=4,
    label="SGD with Momentum Path",
)
plt.plot(
    history_nomomentum_x,
    history_nomomentum_loss,
    color="orange",
    marker="s",
    linestyle=":",
    markersize=4,
    label="SGD without Momentum Path",
)

plt.title("SGD Paths on a Loss Function with a Local Minimum (Using PyTorch API)")
plt.xlabel("Parameter (x)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.ylim(bottom=-1.0, top=5)  # Adjust Y-axis limits for better visualization
plt.show()
