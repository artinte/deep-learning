import torch
import math


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
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
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class MultiheadAttentionScratch(torch.nn.Module):
    """
    A scratch implementation of the Multi-Head Attention mechanism.

    This module performs the following steps:
    1. Projects the input tensors (query, key, value) into different subspaces.
    2. Splits these projections into multiple 'heads'.
    3. Computes and combines attention and padding masks.
    4. For each head, it computes scaled dot-product attention using the
       optimized torch.nn.functional.scaled_dot_product_attention.
    5. Concatenates the outputs from all heads.
    6. Applies a final linear projection to get the final output.
    """

    def __init__(self, d_model, num_heads, dropout=0.0, batch_first=False):
        super(MultiheadAttentionScratch, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(
        self, query, key, value, key_padding_mask=None, attn_mask=None, is_causal=False
    ):
        if self.batch_first:
            batch_size, seq_len, _ = query.size()
        else:
            seq_len, batch_size, _ = query.size()

        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)

        if self.batch_first:
            # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, head_dim)
            q_proj = q_proj.view(
                batch_size, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            k_proj = k_proj.view(
                batch_size, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            v_proj = v_proj.view(
                batch_size, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)
        else:
            # (batch_size, num_heads, seq_len, head_dim)
            q_proj = (
                q_proj.transpose(0, 1)
                .view(batch_size, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k_proj = (
                k_proj.transpose(0, 1)
                .view(batch_size, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            v_proj = (
                v_proj.transpose(0, 1)
                .view(batch_size, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        if key_padding_mask is not None:
            if attn_mask is None:
                attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            else:
                attn_mask = attn_mask.logical_or(
                    key_padding_mask.unsqueeze(1).unsqueeze(1)
                )

        context = torch.nn.functional.scaled_dot_product_attention(
            q_proj,
            k_proj,
            v_proj,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # (batch_size, seq_len, num_heads, head_dim)
        output = context.transpose(1, 2).contiguous()

        if self.batch_first:
            output = output.view(batch_size, seq_len, self.d_model)
        else:
            output = output.view(batch_size, seq_len, self.d_model).transpose(0, 1)

        return self.dropout_layer(self.out_proj(output))
