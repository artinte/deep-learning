import sys
import torch
import torchvision

print("Python version:", sys.version)
print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)


if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
elif torch.backends.mps.is_available():
    print("MPS version:", torch.backends.mps.version)


rand_tensor = torch.rand(size=(5, 3))
print("Random tensor:", rand_tensor)
print("Tensor shape:", rand_tensor.shape)
print("Tensor dtype:", rand_tensor.dtype)
print("Tensor device:", rand_tensor.device)
