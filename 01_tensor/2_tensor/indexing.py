import torch

tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

assert tensor2d[0, 0] == 1
assert tensor2d[1, 2] == 6
assert tensor2d[-1, -1] == 9

tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
assert (tensor2d[0] == torch.tensor([1, 2, 3])).all()
assert (tensor2d[:, 1] == torch.tensor([2, 5, 8])).all()
assert (tensor2d[1:, 1:] == torch.tensor([[5, 6], [8, 9]])).all()

tensor1d = torch.tensor([10, 20, 30, 40, 50])
tensor_mask = tensor1d > 25
assert (tensor_mask == torch.tensor([False, False, True, True, True])).all()
assert (tensor1d[tensor_mask] == torch.tensor([30, 40, 50])).all()

tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_indices = torch.tensor([0, 1, 2])
col_indices = torch.tensor([2, 1, 0])
assert (tensor2d[row_indices, col_indices] == torch.tensor([3, 5, 7])).all()

assert (tensor2d[..., 0] == torch.tensor([1, 4, 7])).all()

tensor1d = torch.tensor([1, 2, 3])
assert tensor1d.shape == (3,)
tensor2d = tensor1d[:, None]
assert tensor2d.shape == (3, 1)
