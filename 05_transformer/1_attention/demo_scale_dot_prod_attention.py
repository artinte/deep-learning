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

torch.manual_seed(0)
# (batch_size, num_heads, seq_len, head_dim)
query = torch.rand(32, 8, 128, 64)
key = torch.rand(32, 8, 128, 64)
value = torch.rand(32, 8, 128, 64)

# Test the official PyTorch implementation without any masking
output_torch = torch.nn.functional.scaled_dot_product_attention(query, key, value)
assert output_torch.shape == (32, 8, 128, 64)

# Test the custom, "from-scratch" implementation
output_scratch = scaled_dot_product_attention(query, key, value)
assert output_scratch.shape == (32, 8, 128, 64)

# Verify that the outputs from both implementations are numerically identical
assert torch.allclose(output_torch, output_scratch, atol=1e-5)

# Test the official PyTorch implementation using the built-in causal flag
output_torch_causal = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
output_scratch_causal = scaled_dot_product_attention(query, key, value, is_causal=True)
assert torch.allclose(output_torch_causal, output_scratch_causal, atol=1e-5)

seq_len = query.size(-2)
# `torch.triu(..., diagonal=1)` creates a mask with `True` values on and above the main diagonal.
# `.logical_not()` inverts this, creating a lower-triangular mask with `True` values
# below the diagonal, indicating valid attention paths.
causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).logical_not()
output_torch_mask = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=causal_mask)
assert torch.allclose(output_torch_causal, output_torch_mask, atol=1e-5)

batch_size = query.size(0)
key_padding_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=bool)
attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(1)
print(attn_mask.shape)
output_torch = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
