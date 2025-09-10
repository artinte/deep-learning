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
output_torch = torch.nn.functional.scaled_dot_product_attention(query, key, value)
assert output_torch.shape == (32, 8, 128, 64)
output_scratch = scaled_dot_product_attention(query, key, value)
assert output_scratch.shape == (32, 8, 128, 64)
assert torch.allclose(output_torch, output_scratch, atol=1e-5)

query = [["to", "be", "or", "not", "to", "be"],
         ["that", "is", "a", "question", "<pad>", "<pad>"],
         ["whether", "tis", "<pad>","<pad>","<pad>","<pad>"],
         ["nobler", "in", "the", "<pad>", "<pad>", "<pad>"]]
all_words = [word for sentence in query for word in sentence]
unique_words = set(all_words)
# index of <pad> is 0
vocab_list = ["<pad>"] if "<pad>" in unique_words else []
vocab_list += [word for word in unique_words if word != "<pad>"]
vocab = {word: idx for idx, word in enumerate(vocab_list)}
indexed_query = []
for sentence in query:
    indexed_sentence = [vocab[word] for word in sentence]
    indexed_query.append(indexed_sentence)

query_tensor = torch.tensor(indexed_query, dtype=torch.long)
assert query_tensor.shape == (4, 6)
print(query_tensor)

vocab_size = len(vocab)
embed_dim = 64
num_heads = 8
head_dim = embed_dim // num_heads

embedding = torch.nn.Embedding(vocab_size, embed_dim)
qkv_proj = torch.nn.Linear(embed_dim, 3 * embed_dim)
out_proj = torch.nn.Linear(embed_dim, embed_dim)

def self_attention(query):
    batch_size, seq_len = query.shape
    
    # [batch_size, 1, 1, seq_len]
    pad_mask = (query == 0).unsqueeze(1).unsqueeze(2)
    
    # [batch_size, seq_len, embed_dim]
    x = embedding(query)
    # [batch_size, seq_len, embed_dim * 3]
    qkv = qkv_proj(x)
    q, k, v = torch.split(qkv, embed_dim, dim=-1)
    
    # [batch_size, num_heads, seq_len, head_dim]
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    output = out_proj(attn_output)
    return output

output = self_attention(query_tensor)
print(output.shape)
