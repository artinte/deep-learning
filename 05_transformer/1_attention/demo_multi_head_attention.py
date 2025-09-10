import torch

torch.manual_seed(0)

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

# [batch_size, 1, 1, seq_len]
pad_mask = (query_tensor == 0).unsqueeze(1).unsqueeze(2)
query_embed = embedding(query_tensor)
multi_head_attention = torch.nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multi_head_attention(query_embed, query_embed, query_embed)

def multihead_self_attention(query):
    batch_size, seq_len = query.shape

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

output_scratch = multihead_self_attention(query_tensor)
print(output_scratch.shape)
assert attn_output.shape == output_scratch.shape
