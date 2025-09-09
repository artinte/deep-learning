import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 10 # Increased epochs for better convergence, feel free to adjust
LR = 0.001
IMAGE_SIZE = 28
PATCH_SIZE = 7  # Each patch is 7x7
NUM_CLASSES = 10
DIM = 64
DEPTH = 6
HEADS = 8
MLP_DIM = 128

# Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class PatchEmbedding(torch.nn.Module):
    def __init__(self, img_size, patch_size, in_channels=1, dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = torch.nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, dim]
        return x

class ViT(torch.nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10, dim=64, depth=6, heads=8, mlp_dim=128):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = torch.nn.Parameter(torch.randn(1, num_patches + 1, dim))

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=False) # Add batch_first=False for clarity (it's the default)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, dim]
        x = x + self.pos_embed

        # --- CRITICAL CHANGE ---
        # Transformer expects (sequence_length, batch_size, embedding_dimension)
        x = x.transpose(0, 1)  # Change from [B, N+1, dim] to [N+1, B, dim]

        x = self.transformer(x)  # [N+1, B, dim]

        # Transpose back to [B, N+1, dim] before taking cls_token
        x = x.transpose(0, 1)

        cls_out = x[:, 0]
        return self.mlp_head(cls_out)

def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

model = ViT().to(device)
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# Training
print("Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {100*correct/total:.2f}%")

# Testing
print("\nStarting Testing...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"Test Accuracy: {100*correct/total:.2f}%")
