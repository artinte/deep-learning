import glob
import os
from matplotlib import pyplot
import soundfile

data_dir = 'data/free-spoken-digit-dataset/recordings'

wav_files = glob.glob(os.path.join(data_dir, '*.wav'))

samples = wav_files[:4]

fig, axs = pyplot.subplots(2, 2, figsize=(10, 6))
axs = axs.flatten()

for idx, f in enumerate(samples):
    audio, sr = soundfile.read(f, dtype='float32')
    axs[idx].plot(audio)
    axs[idx].set_title(f"{os.path.basename(f)}")

pyplot.tight_layout()
pyplot.show()
