import sys
import pathlib
import os
import tarfile

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import download

url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

file_path = download.download(url, sha1_hash='01ada507287d82875905620988597833ad4e0903')

extract_dir = os.path.dirname(file_path)
os.makedirs(extract_dir, exist_ok=True)

with tarfile.open(file_path, 'r:gz') as tar:
    tar.extractall(path=extract_dir)

extract_file_dir = os.path.join(extract_dir, 'aclImdb')
print('Successfully extracted to:', extract_file_dir)

train_dir = os.path.join(extract_file_dir, 'train')
print(os.listdir(train_dir))

train_pos_dir = os.path.join(extract_file_dir, 'train/pos')
train_neg_dir = os.path.join(extract_file_dir, 'train/neg')
print('Train positive dir:', train_pos_dir)
print('Train negative dir:', train_neg_dir)

train_pos_files = os.listdir(train_pos_dir)
train_neg_files = os.listdir(train_neg_dir)

examples_files = train_pos_files[:3]
for index, fname in enumerate(examples_files):
    fpath = os.path.join(train_pos_dir, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
        print(str(index + 1) + ':', content)
