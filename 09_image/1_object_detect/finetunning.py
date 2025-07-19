import sys
import pathlib
import zipfile
import os
import random
import torchvision
from matplotlib import pyplot

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import download

url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'


file_path = download.download(url,
                              sha1_hash='88474aa75cc41dbb8d3c76d2f3c818e79fa0438d')
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

image = torchvision.io.read_image(os.path.join(extract_dir, 'PNGImages/FudanPed00046.png'))
mask = torchvision.io.read_image(os.path.join(extract_dir, 'PedMasks/FudanPed00046_mask.png'))

pyplot.figure(figsize=(8, 4))
pyplot.subplot(121)
pyplot.title('Image')
pyplot.imshow(image.permute(1, 2, 0))
pyplot.subplot(122)
pyplot.title('Mask')
pyplot.imshow(mask.permute(1, 2, 0))
pyplot.show()
