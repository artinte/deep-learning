import torch

tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])

assert tensor2d[0, 0] == 1
assert tensor2d[1, 2] == 6
assert tensor2d[-1, -1] == 9
