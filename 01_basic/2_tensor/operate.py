import torch

ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5
assert torch.isclose(ones, torch.tensor([[1.0, 1.0], [1.0, 1.0]])).all()
assert torch.isclose(twos, torch.tensor([[2.0, 2.0], [2.0, 2.0]])).all()
assert torch.isclose(threes, torch.tensor([[3.0, 3.0], [3.0, 3.0]])).all()
assert torch.isclose(fours, torch.tensor([[4.0, 4.0], [4.0, 4.0]])).all()
print(sqrt2s)

fives = ones + fours
assert torch.isclose(fives, torch.tensor([[5.0, 5.0], [5.0, 5.0]])).all()

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print(torch.mm(a, b))
assert (torch.mm(a, b) == a @ b).all()

a = torch.tensor([1, 2, 3, 4])
assert a.sum() == 10

a = torch.tensor([[1, 2], [3, 4]])
assert (a.sum(dim=0) == torch.tensor([4, 6])).all()

a = torch.tensor([1.0, 2.0, 3.0, 4.0])
assert torch.isclose(a.mean(), torch.tensor(2.5))
assert torch.isclose(a.max(), torch.tensor(4.0))
assert torch.isclose(a.min(), torch.tensor(1.0))

a = torch.tensor([1, 2, 3])
b = torch.tensor([[1], [2], [3]])
c = a + b
assert a.shape == (3,)
assert b.shape == (3, 1)
assert c.shape == (3, 3)
print(c)

a = torch.randn((2, 3, 4))
b = a.view((6, 4))
c = a.view((-1, 4))
d = a.reshape((6, 4))
assert a.shape == (2, 3, 4)
assert b.shape == (6, 4)
assert c.shape == (6, 4)
assert a.data_ptr() == d.data_ptr()

a = torch.randn((2, 3))
b = a.transpose(0, 1)
c = torch.randn((2, 3, 4))
d = c.transpose(1, 2)
assert b.shape == (3, 2)
assert d.shape == (2, 4, 3)

a = torch.randn((1, 3, 1, 4))
b = a.squeeze()
c = torch.randn((1, 3, 1, 4))
d = c.squeeze(0)
assert b.shape == (3, 4)
assert d.shape == (3, 1, 4)

a = torch.randn((3, 4))
b = a.unsqueeze(0)
assert b.shape == (1, 3, 4)

a = torch.randn((2, 3, 4))
b = a.permute(2, 0, 1)
assert b.shape == (4, 2, 3)

a = torch.randn((2, 3, 4))
b = a.flatten()
c = a.flatten(start_dim=1)
assert b.shape == (24,)
assert c.shape == (2, 12)
