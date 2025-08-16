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


n_head = 2
q = q.view(batch_size, seq_length, n_head, n_embed // n_head).transpose(1, 2)
k = k.view(batch_size, seq_length, n_head, n_embed // n_head).transpose(1, 2)
v = v.view(batch_size, seq_length, n_head, n_embed // n_head).transpose(1, 2)
print('Query divided shape:', q.shape)
print('Key divided shape:', k.shape)
print('Value divied shape:', v.shape)

y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
z = y.transpose(1, 2).contiguous().view(batch_size, seq_length, n_embed)
print('Result of scaled_dot_product_attention:', y.shape)
print('Re-assemble of result:', z.shape)
