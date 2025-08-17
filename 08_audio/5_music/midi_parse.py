import pathlib
import sys
import zipfile

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import download


midi_path = "https://github.com/artinte/tiny-datasets/raw/refs/heads/develop/MIDI.zip"

filepath = download.download(midi_path)
print(filepath)

try:
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall('data/')
except zipfile.BadZipFile:
    print('Unzip error')


