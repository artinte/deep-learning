import torchvision

resnet34 = torchvision.models.resnet34(pretrained=True)
print(resnet34)
