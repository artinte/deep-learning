import torch

# Custom Multi-Head Attention layer using F.scaled_dot_product_attention
class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0.1, batch_first: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
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
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ):

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention.
        if self.batch_first:
            q = q.view(-1, q.size(1), self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(-1, k.size(1), self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(-1, v.size(1), self.nhead, self.head_dim).transpose(1, 2)
        else:
            q = q.view(q.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(k.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(v.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)

        # Pass the masks to scaled_dot_product_attention.
        # F.scaled_dot_product_attention automatically handles key_padding_mask.
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
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(-1, query.size(1), self.d_model)
            )
        else:
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(query.size(0), -1, self.d_model)
            )

        output = self.out_proj(attn_output)
        return output, None
