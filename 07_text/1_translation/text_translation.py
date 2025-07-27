import datasets
import transformers
import torch
import random
from matplotlib import pyplot

# 1. Load the dataset
dataset = datasets.load_dataset('ted_hrlr', 'pt_to_en', trust_remote_code=True)
train_examples = dataset['train']
val_examples = dataset['validation']
test_examles = dataset['test']

print(f'Train examples: {len(train_examples)}')
print(f'Validation examples: {len(val_examples)}')
print(f'Test examples: {len(test_examles)}')

print(train_examples[0])

# 2. Initialize a tokenizer
# We'll use a pre-trained tokenizer suitable for sequence-to-sequence tasks (like translation).
# 'Helsinki-NLP/opus-mt-pt-en' is a good choice for Portuguese to English.
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/m2m100_418M")
tokenizer.src_lang = 'pt'
tokenizer.tgt_lang = 'en'

text = 'Hello, how are you?'
encoded = tokenizer(text, return_tensors='pt')
print('Token IDs:', encoded['input_ids'])
print('Tokens:', tokenizer.convert_ids_to_tokens(encoded['input_ids'][0]))


# 3. Create a custom PyTorch Dataset class
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        pair = self.examples[idx]['translation']
        source_text = pair['pt']
        target_text = pair['en']
        
        # Tokenize source and target texts
        # add_special_tokens=True adds [CLS] and [SEP] tokens
        # truncation=True truncates sequences longer than max_length
        # padding='max_length' pads sequences shorter than max_length
        # return_tensors='pt' returns PyTorch tensors
        tokenized_source = self.tokenizer(
            source_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        tokenized_target = self.tokenizer(
            target_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Remove the batch dimension (squeeze) as __getitem__ expects single examples
        return {
            'input_ids': tokenized_source['input_ids'].squeeze(0),
            'attention_mask': tokenized_source['attention_mask'].squeeze(0),
            # For translation, target input_ids are often used as labels
            'labels': tokenized_target['input_ids'].squeeze(0) 
        }
        
# 4. Instantiate your PyTorch datasets
max_seq_length = 128 # You can adjust this based on your data and model
train_dataset_pt = TranslationDataset(train_examples, tokenizer, max_length=max_seq_length)
val_dataset_pt = TranslationDataset(val_examples, tokenizer, max_length=max_seq_length)

print(f"PyTorch Training Dataset size: {len(train_dataset_pt)}")
print(f"PyTorch Validation Dataset size: {len(val_dataset_pt)}")

# Example of accessing an item from the PyTorch dataset
sample_idx = random.randint(0, 100)
sample_item = train_dataset_pt[sample_idx]
print("Sample item from PyTorch training dataset:")
print(f"Input IDs shape: {sample_item['input_ids'].shape}")
print(f"Attention Mask shape: {sample_item['attention_mask'].shape}")
print(f"Labels shape: {sample_item['labels'].shape}")

original_src_text = train_examples[sample_idx]['translation']['pt']
print(f"Original Source Text (PT): {original_src_text}")

src_token_ids = sample_item['input_ids'].tolist()
print(f"Source Token IDs: {src_token_ids}")

attention_mask_values = sample_item['attention_mask'].tolist()
print(f'Attention  Mask Values: {attention_mask_values}')

tgt_token_ids = sample_item['labels'].tolist()
print(f"Target (Label) Token IDs: {tgt_token_ids}")

# Convert source token IDs to text.
src_text = tokenizer.decode(src_token_ids, skip_special_tokens=True)
print(f'Source Text: {src_text}')
tgt_text = tokenizer.decode(tgt_token_ids, skip_special_tokens=True)
print(f'Target (Label) Text: {tgt_text}')

train_dataloader = torch.utils.data.DataLoader(train_dataset_pt, batch_size=16, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset_pt, batch_size=16, shuffle=False)

print(f"Number of batches in training DataLoader: {len(train_dataloader)}")
print(f"Number of batches in validation DataLoader: {len(val_dataloader)}")

# Example of iterating through a batch
for batch in train_dataloader:
    print("Sample batch from DataLoader:")
    print(f"Input IDs batch shape: {batch['input_ids'].shape}")
    print(f"Attention Mask batch shape: {batch['attention_mask'].shape}")
    print(f"Labels batch shape: {batch['labels'].shape}")
    break

def positional_encoding(length, depth):
    depth = depth / 2
    
    # (seq, 1)
    positions = torch.arange(length).unsqueeze(1)
    # (1, depth)
    depths = torch.arange(int(depth)).unsqueeze(0) / depth
    # (1, depth)
    angle_rates = 1 / (10000 ** depths)
    # (pos, depth)
    angle_rads = positions * angle_rates
    
    pos_encoding = torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], axis=-1)
    return pos_encoding.float()

    
pos_encoding = positional_encoding(length=2048, depth=512)
assert pos_encoding.shape == (2048, 512)
# Plot the dimensions.
pyplot.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
pyplot.xlabel('Position')
pyplot.ylabel('Depth')
pyplot.colorbar()
pyplot.show()
