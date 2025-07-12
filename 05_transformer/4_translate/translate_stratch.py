import torch
import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from common import make_model, subsequent_mask

def train_worker(gpu, ngpus_per_node,
                 vocab_src, vocab_tgt):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)