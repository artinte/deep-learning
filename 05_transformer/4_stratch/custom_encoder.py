import torch
from multi_head_attention import MultiHeadAttention


# Custom Transformer Encoder Layer using the new attention mechanism
class CustomEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        batch_first=True,
    ):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        if activation == "relu":
            self.activation = torch.nn.functional.relu
        elif activation == "gelu":
            self.activation = torch.nn.functional.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self-attention block with is_causal=False
        x = src
        x = x + self.dropout1(
            self.self_attn(
                x,
                x,
                x,
                key_padding_mask=src_key_padding_mask,
                attn_mask=src_mask,
                is_causal=is_causal,
            )[0]
        )
        x = self.norm1(x)

        # Feedforward block
        x = x + self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )
        x = self.norm2(x)

        return x


# Custom Transformer Encoder that stacks the custom layers
class CustomEncoder(torch.nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        return output
