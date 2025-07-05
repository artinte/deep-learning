import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import os
from matplotlib import pyplot
import numpy as np
from PIL import Image
import torch


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if needed (to match x2's size due to cropping in original U-Net)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class OxfordPetsDataset(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.dataset = OxfordIIITPet(
            root=root, download=True, target_types='segmentation')
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.indices = list(range(len(self.dataset)))
        split_point = int(0.8 * len(self.dataset))
        self.indices = self.indices[:split_point] if split == 'train' else self.indices[split_point:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self.dataset[self.indices[idx]]

        # Segmentation mask: 1 = pet, 2 = border, 0 = background
        # We'll convert to binary mask: pet (1), background/border (0)
        mask = np.array(mask)
        mask = (mask == 1).astype(np.float32)
        mask = Image.fromarray(mask)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        return img, mask


image_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mask_transform = T.Compose([
    T.Resize((128, 128), interpolation=Image.NEAREST),
    T.ToTensor()  # keep it [0, 1]
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = OxfordPetsDataset(root='./data', split='train',
                                  transform=image_transform,
                                  target_transform=mask_transform)
val_dataset = OxfordPetsDataset(root='./data', split='val',
                                transform=image_transform,
                                target_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

model = UNet(in_channels=3, num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

NUM_EPOCHS = 5


def train_epoch():
    model.train()
    total_loss = 0
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def eval_epoch():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            total_loss += loss.item()
    return total_loss / len(val_loader)


for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch()
    val_loss = eval_epoch()
    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


model.eval()
img, mask = val_dataset[0]
with torch.no_grad():
    pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))
    pred_mask = (pred > 0.5).float().cpu().squeeze()

pyplot.figure(figsize=(12, 4))
pyplot.subplot(1, 3, 1)
pyplot.title("Input Image")
pyplot.imshow(img.permute(1, 2, 0))

pyplot.subplot(1, 3, 2)
pyplot.title("Ground Truth")
pyplot.imshow(mask.squeeze(), cmap='gray')

pyplot.subplot(1, 3, 3)
pyplot.title("Prediction")
pyplot.imshow(pred_mask, cmap='gray')
pyplot.show()
