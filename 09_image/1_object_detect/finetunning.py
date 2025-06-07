import sys
import pathlib
import zipfile
import os
import random
from matplotlib import pyplot

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import download

url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'



file_path = download.download(url,
                              sha1_hash='88474aa75cc41dbb8d3c76d2f3c818e79fa0438d')
print(file_path)
