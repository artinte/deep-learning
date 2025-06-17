import random
import sys

import kagglehub
import os
import torchvision
import json
import pathlib
import torch
from PIL import Image

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from data.conv_mlp.src.convmlp import convmlp_s

# https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset

class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))], key=lambda x: int(x))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for image_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, image_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

if __name__ == '__main__':
    # Download latest version
    path = kagglehub.dataset_download("nunenuh/pytorch-challange-flower-dataset")

    print("Path to dataset files:", path)

    train_dir = os.path.join(path, 'dataset/train')
    valid_dir = os.path.join(path, 'dataset/valid')
    test_dir = os.path.join(path, 'dataset/test')

    with open(os.path.join(path, 'cat_to_name.json'), 'r') as f:
        class_names = json.load(f)

    batch_size = 32
    img_size = (150, 150)

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        torchvision.transforms.ToTensor(),
    ])

    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.ToTensor(),
    ])

    train_dataset = CustomImageFolder(
        root_dir=train_dir, transform=transform_train)
    train_loader = torch.utils.data.dataloader.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomImageFolder(
        root_dir=valid_dir, transform=transform_val)
    val_loader = torch.utils.data.dataloader.DataLoader(
        val_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_sm = convmlp_s(pretrained=False,
                         progress=True,
                         num_classes=len(class_names))
    model_sm = model_sm.to(device)

    optimizer = torch.optim.Adam(model_sm.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 100
    for epoch in range(epochs):
        model_sm.train()
        train_loss = 0.0
        correct, total, acc = 0, 0, 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_sm(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # update model parameters

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 10 == 0:
                acc = 100 * correct / total  # calculate accuracy
                sys.stdout.write(f'\rEpoch [{epoch + 1}/{epochs}], step [{i + 1}/{len(train_loader)}], '
                                 f'loss: {train_loss / (i + 1):.4f}, acc: {acc:.2f}%')
                sys.stdout.flush()
        print()
        avg_loss = train_loss / len(train_loader)
        avg_acc = 100 * correct / total

        model_sm.eval()
        val_loss = 0
        val_acc = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                output = model_sm(data)
                val_total += target.size(0)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100 * val_acc / val_total
        print(f'Epoch [{epoch + 1}/{epochs}], loss: {avg_loss:.4f}, acc: {avg_acc:.2f}%, '
              f'valid loss {val_loss:.4f}, valid acc {val_acc:.2f}%')

    # save model
    model_path = 'flower_conv_mlp.pth'
    torch.save(model_sm, model_path)
    print('Save model to', os.path.join(os.getcwd(), model_path))

    subdirs = ['54', '43', '32']

    model = torch.load(model_path, weights_only=False)
    model = model.to(device)
    model.eval()  # 评估模式

    for subdir in subdirs:
        image_files = [f for f in os.listdir(os.path.join(
            train_dir, subdir)) if f.endswith('.jpg') or f.endswith('.png')]
        random_image = random.choice(image_files)
        image = Image.open(os.path.join(os.path.join(train_dir, subdir), random_image))

        input_tensor = transform_val(image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            input_batch = input_batch.to(device)
            output = model(input_batch)

        _, predicted_class = torch.max(output, 1)
        key_flower = str(predicted_class.item() + 1)
        print(random_image, key_flower, class_names[key_flower])

    # save onnx
    device = torch.device('cpu')
    model_sm.to(device)
    model_sm.eval()
    dummy_input = torch.randn(1, 3, *img_size)
    onnx_model_path = 'flower_conv_mlp.onnx'
    torch.onnx.export(
        model_sm,
        dummy_input,
        onnx_model_path,
        do_constant_folding=True,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=18,
    )
