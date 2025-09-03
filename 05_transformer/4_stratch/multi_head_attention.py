import torch


# Custom Multi-Head Attention using torch.nn.functional.scaled_dot_product_attention
class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0.1, batch_first: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.head_dim = d_model // nhead
        self.batch_first = batch_first

        # Linear layers for projecting query, key, and value
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        self.dropout_p = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ):
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention.
        if self.batch_first:
            # (batch_size, seq_len, d_model) -> (batch_size, nhead, seq_len, head_dim)
            q = q.view(q.size(0), q.size(1), self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(k.size(0), k.size(1), self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(v.size(0), v.size(1), self.nhead, self.head_dim).transpose(1, 2)
        else:
            # (seq_len, batch_size, d_model) -> (batch_size, nhead, seq_len, head_dim)
            q = q.view(q.size(0), q.size(1), self.nhead, self.head_dim).permute(
                1, 2, 0, 3
            )
            k = k.view(k.size(0), k.size(1), self.nhead, self.head_dim).permute(
                1, 2, 0, 3
            )
            v = v.view(v.size(0), v.size(1), self.nhead, self.head_dim).permute(
                1, 2, 0, 3
            )

        if key_padding_mask is not None:
            # Reshape key_padding_mask from (batch_size, seq_len)
            # to (batch_size, 1, 1, seq_len) for broadcasting with attention scores.
            # unsqueeze(1) adds a dimension for nhead.
            # unsqueeze(2) adds a dimension for query sequence length.
            padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)

            if attn_mask is None:
                # If no other mask exists, create the attention mask from the padding mask.
                attn_mask = padding_mask_expanded
            else:
                # If an existing attention mask (e.g., causal mask) exists,
                # combine the two by performing a logical OR.
                attn_mask = torch.logical_or(attn_mask, padding_mask_expanded)

        # Pass the masks to scaled_dot_product_attention.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape back to the original d_model dimension
        if self.batch_first:
            # (batch_size, nhead, seq_len, head_dim) -> (batch_size, seq_len, d_model)
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(query.size(0), query.size(1), self.d_model)
            )
        else:
            # (batch_size, nhead, seq_len, head_dim) -> (seq_len, batch_size, d_model)
            attn_output = (
                attn_output.permute(2, 0, 1, 3)
                .contiguous()
                .view(query.size(0), query.size(1), self.d_model)
            )

        output = self.out_proj(attn_output)
        return output, None
