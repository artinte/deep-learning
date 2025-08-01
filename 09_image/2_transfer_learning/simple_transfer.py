import pathlib
import sys
import os
import numpy
import torch
import zipfile
import util
from PIL import Image
from matplotlib import pyplot
import torchvision
from torchvision import datasets, transforms

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import download

data_url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = numpy.array([0.485, 0.456, 0.406])
    std = numpy.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = numpy.clip(inp, 0, 1)
    pyplot.imshow(inp)
    if title is not None:
        pyplot.title(title)
    pyplot.pause(0.001)  # pause a bit so that plots are updated
    pyplot.show()


def visualize_model(model):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        inputs, labels = next(iter(dataloaders["val"]))
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        _, axes = pyplot.subplots(2, 2)
        axes = axes.flatten()
        for j in range(2 * 2):
            ax = axes[j]
            ax.axis("off")
            ax.set_title(f"predicted: {class_names[preds[j]]}")
            image_to_show = inputs.cpu()[j].permute(1, 2, 0).numpy()
            mean = numpy.array([0.485, 0.456, 0.406])
            std = numpy.array([0.229, 0.224, 0.225])
            image_to_show = image_to_show * std + mean
            image_to_show = numpy.clip(image_to_show, 0, 1)
            ax.imshow(image_to_show)

    # Restore model to original training state
    model.train(mode=was_training)
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    file_path = download.download(
        data_url, sha1_hash="c6ff178de032ee56fa5b35c2a6a9005c934faba0"
    )
    print(file_path)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    extract_dir = os.path.join(os.path.dirname(file_path), base_name)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        names = zip_ref.namelist()
        top_level_dirs = {name.split("/")[0] for name in names if "/" in name}
        if len(top_level_dirs) == 1 and base_name in top_level_dirs:
            zip_ref.extractall(os.path.dirname(file_path))
            os.rename(os.path.join(os.path.dirname(file_path), base_name), extract_dir)
        else:
            os.makedirs(extract_dir, exist_ok=True)
            zip_ref.extractall(extract_dir)

    # print the contents of the extracted directory
    for item in os.listdir(extract_dir):
        item_path = os.path.join(extract_dir, item)
        if os.path.isdir(item_path):
            print(f"Directory: {item_path}")
        else:
            print(f"File: {item_path}")

    # Load the data
    print("Loading data...")
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(extract_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=4, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Class names: {class_names}")

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders["train"]))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

    print("Fine-tuning the convnet...")
    model_ft = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = torch.nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1
    )

    model_ft = util.train_model(
        model_ft,
        dataloaders,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        device,
        num_epochs=25,
    )
    visualize_model(model_ft)

    print("ConvNet as fixed feature extractor...")
    model_conv = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = torch.nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = torch.optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_conv, step_size=7, gamma=0.1
    )

    model_conv = util.train_model(
        model_conv,
        dataloaders,
        criterion,
        optimizer_conv,
        exp_lr_scheduler,
        device,
        num_epochs=25,
    )
    visualize_model(model_conv)

    # Visualize predictions on custom images
    img_path = "data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg"
    img = Image.open(img_path)
    img = data_transforms["val"](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    model_conv.eval()
    with torch.no_grad():
        outputs = model_conv(img)
        _, preds = torch.max(outputs, 1)

    predicted_class = class_names[preds[0]]
    print(f"Predicted class for the image {img_path}: {predicted_class}")
