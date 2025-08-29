import torch
import math


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        # head dimension
        self.d_k = d_model // n_heads
        # head number
        self.n_heads = n_heads

        # Linear projections for Query, Key, and Value
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)

        # Final linear layer after concatenating heads
        self.out_linear = torch.nn.Linear(d_model, d_model)
        self.dropout_rate = dropout

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        """
        Args:
            query, key, value (Tensor): Tensors of shape (batch_size, seq_len, d_model).
            attn_mask (Tensor): A boolean mask of shape (batch_size, 1, 1, seq_len) or (batch_size, seq_len, seq_len)
                                or (1, 1, seq_len, seq_len), where True indicates positions to be masked.
            is_causal (bool): If True, a causal mask is automatically applied.
        """
        batch_size = query.size(0)

        # Linear projections and reshaping for multi-head attention
        # Shape: (batch_size, n_heads, seq_len, d_k)
        query = (
            self.q_linear(query)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        key = (
            self.k_linear(key)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        value = (
            self.v_linear(value)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Call the highly optimized scaled_dot_product_attention function
        # This single function replaces the manual implementation of scaling, softmax, and dropout
        # The attn_mask parameter should be passed in a way that aligns with PyTorch's
        # internal conventions for the function.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape and pass through final linear layer
        # Shape: (batch_size, seq_len, d_model)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_k)
        )
        output = self.out_linear(attn_output)

        return output
