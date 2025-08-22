import math
import torch

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


batch_size = 2
num_heads = 4
seq_len = 10
embed_dim = 16

query = torch.randn(batch_size, num_heads, seq_len, embed_dim)
key = torch.randn(batch_size, num_heads, seq_len, embed_dim)
value = torch.randn(batch_size, num_heads, seq_len, embed_dim)

output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
print(f'Shape of the output tensor: {output.shape}')

# specify a mask
attn_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
output_with_mask = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
print(f'Shape of the output tensor with mask: {output_with_mask.shape}')
