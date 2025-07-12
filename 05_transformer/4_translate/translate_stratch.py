import torch

def train_worker(gpu, ngpus_per_node,
                 vocab_src, vocab_tgt):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)