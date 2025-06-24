import torch
import torchvision


class DeepNN(torch.nn.Module):
    # Deeper neural network class to be used as teacher.
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LightNN(torch.nn.Module):
    # Lightweight neural network class to be used as student
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, train_loader, epochs, learning_rate, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)

            # outputs: Output of the network for the collection of images.
            # A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')


def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    return accuracy


if __name__ == '__main__':
    # Check if the current `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`
    # is available, and if not, use the CPU
    device = torch.accelerator.current_accelerator(
    ).type if torch.accelerator.is_available() else "cpu"
    print(f'Using {device} device')

    # Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
    transforms_cifar = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loading the CIFAR-10 dataset:
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transforms_cifar)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transforms_cifar)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2)

    torch.manual_seed(42)
    nn_deep = DeepNN(num_classes=10).to(device)
    train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
    test_accuracy_deep = test(nn_deep, test_loader, device=device)

    # Initialize the lightweight neural network.
    torch.manual_seed(42)
    nn_light = LightNN(num_classes=10).to(device)

    torch.manual_seed(42)
    new_nn_light = LightNN(num_classes=10).to(device)

    # Print the norm of the first layer of the initial lightweight model
    print("Norm of 1st layer of nn_light:", torch.norm(
        nn_light.features[0].weight).item())
    # Print the norm of the first layer of the new lightweight model
    print("Norm of 1st layer of new_nn_light:", torch.norm(
        new_nn_light.features[0].weight).item())

    total_params_deep = "{:,}".format(
        sum(p.numel() for p in nn_deep.parameters()))
    print(f"DeepNN parameters: {total_params_deep}")
    total_params_light = "{:,}".format(
        sum(p.numel() for p in nn_light.parameters()))
    print(f"LightNN parameters: {total_params_light}")

    train(nn_light, train_loader, epochs=10,
          learning_rate=0.001, device=device)
    test_accuracy_light_ce = test(nn_light, test_loader, device)

    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")
