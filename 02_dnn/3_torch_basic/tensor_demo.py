import numpy
import torch

# directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
assert x_data.shape == (2, 2)

# from a numpy array
np_array = numpy.array(data)
x_np = torch.from_numpy(np_array)
assert x_np.shape == (2, 2)

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
