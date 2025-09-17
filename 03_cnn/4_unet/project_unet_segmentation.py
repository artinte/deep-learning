import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import pyplot
from PIL import Image
import torch
from oxford_pets_dataset import OxfordPetsDataset
from unet_model import UNet


def denormalize_image(tensor):
    """
    Denormalizes a PyTorch tensor with the given mean and standard deviation.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


image_transform = T.Compose(
    [
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

mask_transform = T.Compose(
    [T.Resize((128, 128), interpolation=Image.NEAREST), T.ToTensor()]  # keep it [0, 1]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use device: {device}")

train_dataset = OxfordPetsDataset(
    root="./data",
    split="train",
    transform=image_transform,
    target_transform=mask_transform,
)
val_dataset = OxfordPetsDataset(
    root="./data",
    split="val",
    transform=image_transform,
    target_transform=mask_transform,
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)

fig, axes = pyplot.subplots(2, 3)
images, masks = next(iter(train_loader))


for i in range(3):
    de_image = denormalize_image(images[i])
    axes[0, i].imshow(de_image.permute(1, 2, 0).numpy())
    axes[0, i].set_title("Image")
    axes[0, i].axis("off")

    axes[1, i].imshow(masks[i].squeeze().numpy(), cmap="gray")
    axes[1, i].set_title("Mask")
    axes[1, i].axis("off")

pyplot.tight_layout()
pyplot.show()


model = UNet(in_channels=3, num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train():
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


def eval():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            total_loss += loss.item()
    return total_loss / len(val_loader)

epochs = 10
for epoch in range(epochs):
    train_loss = train()
    val_loss = eval()
    print(
        f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )


model.eval()
img, mask = val_dataset[0]
de_img = denormalize_image(img)
with torch.no_grad():
    pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))
    pred_mask = (pred > 0.5).float().cpu().squeeze()

pyplot.subplot(1, 3, 1)
pyplot.title("Input Image")
pyplot.imshow(de_img.permute(1, 2, 0))

pyplot.subplot(1, 3, 2)
pyplot.title("Ground Truth")
pyplot.imshow(mask.squeeze(), cmap="gray")

pyplot.subplot(1, 3, 3)
pyplot.title("Prediction")
pyplot.imshow(pred_mask, cmap="gray")
pyplot.show()
