import torch
import torchvision
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
