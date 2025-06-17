import torch
import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.FashionMNIST(root='./data',
                                              train=True,
                                              download=True,
                                              transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='./data',
                                             train=False,
                                             download=True,
                                             transform=transform)   
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)


class AutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim, shape):
        super(AutoEncoder, latent_dim, shape)
        self.lantent_dim = latent_dim
        self.shape = shape
        self.input_dim = 1
        for dim in shape:
            self.input_dim *= dim
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.input_dim, latent_dim),
            torch.nn.ReLU(),
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, self.input_dim),
            torch.nn.Sigmoid(),
            torch.nn.Unflatten(1, shape)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

latent_dim = 64
autoencoder = AutoEncoder(latent_dim, train_set.data.shape[1:])

sample_input, _ = next(iter(test_loader))
output = autoencoder(sample_input)
print(output.shape)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

for images, _ in train_loader:
    outputs = autoencoder(images)
    loss = criterion(outputs, images)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    