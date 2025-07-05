import math
import torch
from util import masked_softmax, show_heatmaps


print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))

Q = torch.ones((2, 3, 4))
K = torch.ones((2, 4, 6))
assert torch.bmm(Q, K).shape == (2, 3, 6)


class DotProductAttention(torch.nn.Module):
    """Scaled dot product attention."""

    def __init__(self, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


queries = torch.normal(0, 1, (2, 1, 2))
keys = torch.normal(0, 1, (2, 10, 2))
values = torch.normal(0, 1, (2, 10, 4))
valid_lens = torch.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.eval()
assert attention(queries, keys, values, valid_lens).shape == (2, 1, 4)

show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
              xlabel='Keys', ylabel='Queries')
