import glob
import os
import torchaudio
from matplotlib import pyplot
import soundfile

data_dir = "data/free-spoken-digit-dataset/recordings"

wav_files = glob.glob(os.path.join(data_dir, "*.wav"))

samples = wav_files[:4]

fig, axs = pyplot.subplots(2, 2)
axs = axs.flatten()

for idx, f in enumerate(samples):
    audio, sr = soundfile.read(f, dtype="float32")
    axs[idx].plot(audio)
    axs[idx].set_title(f"{os.path.basename(f)}")

pyplot.tight_layout()
pyplot.show()


waveform, sample_rate = torchaudio.load(wav_files[0])

n_fft = 2048
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    n_mels=n_mels,
)
mel_spectrogram = mel_spectrogram_transform(waveform)
mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

mel_spectrogram_db = mel_spectrogram_db.squeeze()
time_bins, mel_bins = mel_spectrogram_db.shape
duration = (time_bins * hop_length) / sample_rate

img = pyplot.imshow(mel_spectrogram_db, aspect='auto', origin='lower', cmap='viridis',
                     extent=[0, duration, 0, n_mels])
pyplot.title('Mel Spectrogram (Torchaudio)')
pyplot.xlabel('Time (s)')
pyplot.ylabel('Mel Frequency Bin')
pyplot.colorbar(img, format="%+2.0f dB")

pyplot.tight_layout()
pyplot.show()