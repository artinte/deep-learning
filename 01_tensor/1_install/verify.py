import torch

print(torch.__version__)

x = torch.rand(size=(5, 3))
print(x)

print(torch.cuda.is_available())
print(torch.mps.is_available())
