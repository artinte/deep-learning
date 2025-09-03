import torch
from multi_head_attention import MultiHeadAttention


# Custom Transformer Decoder Layer using the new attention mechanism
class CustomDecoderLayer(torch.nn.Module):
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
        # self.self_attn = MultiHeadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.cross_attn = torch.nn.MultiheadAttention(
        # self.cross_attn = MultiHeadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        if activation == "relu":
            self.activation = torch.nn.functional.relu
        elif activation == "gelu":
            self.activation = torch.nn.functional.gelu

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=True,
        memory_is_casual=False,
    ):
        # Self-attention block with is_causal=True
        x = tgt
        x = x + self.dropout1(
            self.self_attn(
                x,
                x,
                x,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=tgt_is_causal,
            )[0]
        )
        x = self.norm1(x)

        # Cross-attention block with is_causal=False
        x = x + self.dropout2(
            self.cross_attn(
                x,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                is_causal=memory_is_casual,
            )[0]
        )
        x = self.norm2(x)

        # Feedforward block
        x = x + self.dropout3(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )
        x = self.norm3(x)

        return x


# Custom Transformer Decoder that stacks the custom layers
class CustomDecoder(torch.nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=True,
        memory_is_causal=False,
    ):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_casual=memory_is_causal,
            )
        return output
