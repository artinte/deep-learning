import torch
import torchvision
from PIL import Image
import numpy


class OxfordPetsDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.dataset = torchvision.datasets.OxfordIIITPet(
            root=root, download=True, target_types="segmentation"
        )
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.indices = list(range(len(self.dataset)))
        split_point = int(0.8 * len(self.dataset))
        self.indices = (
            self.indices[:split_point]
            if split == "train"
            else self.indices[split_point:]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self.dataset[self.indices[idx]]

        # Segmentation mask: 1 = pet, 2 = border, 0 = background
        # We'll convert to binary mask: pet (1), background/border (0)
        mask = numpy.array(mask)
        mask = (mask == 1).astype(numpy.float32)
        mask = Image.fromarray(mask)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        return img, mask

