import pathlib
import os
import tarfile
import sys

import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from common import download

def get_dataset_dir():
    '''
    Download the imdb dataset, and return the dataset dir.
    '''
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    file_path = download.download(url, sha1_hash='01ada507287d82875905620988597833ad4e0903')

    extract_dir = os.path.dirname(file_path)
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

    extract_file_dir = os.path.join(extract_dir, 'aclImdb')
    print('Successfully extracted to:', extract_file_dir)
    return extract_file_dir
    train_dir = os.path.join(extract_file_dir, 'train')
    print(os.listdir(train_dir))