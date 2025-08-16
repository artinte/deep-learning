import torch

batch_size = 2
seq_length = 3
n_embed = 4

x = torch.randn((batch_size, seq_length, n_embed))
print('Input tensor shape:', x.shape)

c_attn = torch.nn.Linear(n_embed, 3 * n_embed)
result = c_attn(x)
print('Result shape:', result.shape)
q, k, v = result.split(n_embed, dim=2)
print('Query shape:', q.shape)
print('Key shape:', k.shape)
print('Value shape:', v.shape)
