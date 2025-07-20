import pathlib
import sys
import os
import zipfile
from torchvision import datasets, models, transforms

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import download

data_url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'

file_path = download.download(data_url,
                              sha1_hash='c6ff178de032ee56fa5b35c2a6a9005c934faba0')
print(file_path)

base_name = os.path.splitext(os.path.basename(file_path))[0]
extract_dir = os.path.join(os.path.dirname(file_path), base_name)

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    names = zip_ref.namelist()
    top_level_dirs = {name.split('/')[0] for name in names if '/' in name}
    if len(top_level_dirs) == 1 and base_name in top_level_dirs:
        zip_ref.extractall(os.path.dirname(file_path))
        os.rename(os.path.join(os.path.dirname(file_path), base_name), extract_dir)
    else:
        os.makedirs(extract_dir, exist_ok=True)
        zip_ref.extractall(extract_dir)


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}