import torch
from matplotlib import pyplo

def corrupt(x, noise_amount):
    noise = torch.randn_like(x) * noise_amount.view(-1, 1, 1, 1)
    return x + noise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torchvision.datasets.MNIST(root="data/", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
x, y = next(iter(train_dataloader))

plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
plt.show()


class BasicUNet(nn.Module):
    """最小化的 UNet 实现"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            if i < 2:
                h.append(x)
                x = self.downscale(x)
            
        for i, l in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += h.pop()
            x = self.act(l(x))
            
        return x

net = BasicUNet()
x = torch.rand(8, 1, 28, 28)
print(net(x).shape)
print(sum(p.numel() for p in net.parameters()))

batch_size = 8

n_epochs = 3
net = BasicUNet()
net.to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
losses = []

for epoch in range(n_epochs):
    for x, y in train_dataloader:
        x = x.to(device)
        noise_amount = torch.rand(x.shape[0]).to(device)
        noisy_x = corrupt(x, noise_amount)
        pred = net(noisy_x)
        loss = loss_fn(pred, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f"Avg loss: {avg_loss:.5f}")


plt.plot(losses)
plt.ylim(0, 0.1)
plt.savefig('output/loss_curve.png')
plt.show()

x, y = next(iter(train_dataloader))
x = x[:8]

amount = torch.linspace(0, 1, x.shape[0])
noised_x = corrupt(x, amount)

with torch.no_grad():
    preds = net(noised_x.to(device)).detach().cpu()
