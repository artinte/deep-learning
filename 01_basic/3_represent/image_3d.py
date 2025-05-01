import sys
import pathlib

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import download

url = 'https://github.com/artinte/tiny-datasets/raw/develop/mnist.zip'

file_path = download.download(url, sha1_hash='1f2d5ca7b198c0b293014eb59e9064509d7315b1')
