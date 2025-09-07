import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Define the CNN model using nn.Module
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # A 3x3 convolution with 8 output channels, padding='valid' by default
        # Input: 28x28x1 -> Output: 26x26x8
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        # 2x2 max pooling
        # Input: 26x26x8 -> Output: 13x13x8
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        # A fully connected layer for classification
        # Input: 13*13*8 = 1352 -> Output: 10
        self.fc = torch.nn.Linear(in_features=13 * 13 * 8, out_features=10)

    def forward(self, x):
        # The input x is a batch of images
        x = self.conv(x)
        x = self.pool(x)
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 13 * 13 * 8)
        x = self.fc(x)
        return x


# 1. Data Preparation and Transformations
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # Normalize pixel values to the range [-0.5, 0.5]
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# 2. Set up the DataLoader for batching
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=False
)

# 3. Instantiate the model, loss function, and optimizer
model = SimpleCNN()
criterion =torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# 4. Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (images, labels) in enumerate(train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate loss and accuracy for monitoring
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_predictions / total_samples

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
    )

# 5. Testing loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation for inference
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
