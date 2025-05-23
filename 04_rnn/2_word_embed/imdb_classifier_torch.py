import os
import pathlib
import sys
import shutil

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import imdb_text

dataset_dir = imdb_text.get_dataset_dir()
print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
