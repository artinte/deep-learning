import torch
import numpy

python_list = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(python_list)
print('Tensor from list:', tensor_from_list)

numpy_array = numpy.array([[1, 2], [3, 4]])
tensor_from_array = torch.tensor(numpy_array)
print('Tensor from array:', tensor_from_array)

# avoid a copy
tensor_from_array = torch.as_tensor(numpy_array)

device = 'cuda' if torch.cuda.is_available() else 'mps' \
    if torch.mps.is_available() else 'cpu'
print(device)

tensor_custom = torch.tensor([1, 2, 3],
                             dtype=torch.int64,
                             device=device)
print(tensor_custom)

tensor_zeros = torch.zeros(size=(3, 3))
tensor_ones = torch.ones((2, 2))
tensor_empty = torch.empty((2, 2))
print(tensor_zeros)
print(tensor_ones)
print(tensor_empty)

torch.manual_seed(47)
tensor_rand = torch.rand(2, 2)
tensor_randn = torch.randn(2, 3)
print(tensor_rand)
print(tensor_randn)

tensor_arange = torch.arange(0, 5)
assert (tensor_arange.numpy() == [0, 1, 2, 3, 4]).all()
tensor_linspace = torch.linspace(0, 1, steps=5)
assert torch.allclose(tensor_linspace,
    torch.Tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
