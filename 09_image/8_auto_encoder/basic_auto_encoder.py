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
        super().__init__()
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_dim = 64
autoencoder = AutoEncoder(latent_dim, train_set[0][0].shape).to(device)

sample_input, _ = next(iter(test_loader)).to(device)
output = autoencoder.forward(sample_input)
print(output.shape)


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

for epoch in range(10):
    autoencoder.train()
    total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        outputs = autoencoder(images)
        loss = criterion(outputs, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/10], train Loss: {total_loss / len(train_loader):.4f}")


autoencoder.eval()
mse_total = 0
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = autoencoder(images)
        loss = criterion(outputs, images)
        mse_total += loss.item()

print(f"Test reconstruction MSE: {mse_total / len(test_loader):.4f}")

    