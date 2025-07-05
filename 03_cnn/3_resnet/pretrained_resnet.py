import pathlib
import sys
import torch
import torchvision
from matplotlib import pyplot
from PIL import Image

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import download

resnet = torchvision.models.resnet34(pretrained=True)
resnet.eval()

img_path = download.download('https://img-datasets.s3.amazonaws.com/cat.jpg')
image = Image.open(img_path).convert('RGB')

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]
print(input_tensor.shape)

pyplot.axis('off')
pyplot.imshow(image)
pyplot.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0)
pyplot.show()

imagenet_labels_path = download.download(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
with open(imagenet_labels_path) as f:
    labels = [line.strip() for line in f.readlines()]

with torch.no_grad():
    output = resnet(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top3_prob, top3_idx = torch.topk(probs, 3)

for i in range(3):
    class_idx = top3_idx[i].item()
    prob = top3_prob[i].item()
    label = labels[class_idx]
    print(f"Top {i+1}: {label} ({prob:.4f})")
