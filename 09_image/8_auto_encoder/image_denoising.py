import torch
import torchvision
import torchsummary
from matplotlib import pyplot

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)


sample_images, sample_labels = next(iter(train_loader))
print(sample_images.shape)
print(sample_labels.shape)

noise_factor = 0.2
noisy_images = sample_images + noise_factor * torch.randn_like(sample_images)
noisy_images = torch.clamp(noisy_images, 0.0, 1.0)

n = 10
pyplot.figure(figsize=(6, 2))
for i in range(n):
    ax = pyplot.subplot(1, n, i + 1)
    pyplot.imshow(noisy_images[i].squeeze())
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    pyplot.gray()
pyplot.show()


class Denoise(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                8, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Denoise().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

torchsummary.summary(model.encoder, input_size=(1, 28, 28))
torchsummary.summary(model.decoder, input_size=(8, 7, 7))

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)
        noisy = images + noise_factor * torch.randn_like(images)
        noisy = torch.clamp(noisy, 0.0, 1.0)

        outputs = model(noisy.to(device))
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], loss: {total_loss / len(train_loader):.4f}")


encoded_imgs = model.encoder(sample_images.to(device))
decoded_imgs = model.decoder(encoded_imgs).detach().cpu().numpy()

n = 10
pyplot.figure(figsize=(10, 2))
for i in range(n):
    ax = pyplot.subplot(2, n, i + 1)
    pyplot.imshow(noisy_images[i].squeeze(), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    bx = pyplot.subplot(2, n, i + n + 1)
    pyplot.imshow(decoded_imgs[i].squeeze(), cmap="gray")
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)

pyplot.show()
