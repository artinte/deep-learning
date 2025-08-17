import numpy
import os
from core import audio


def test_audio():
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.aac")
    raw_audio = audio.load_audio(audio_path)
    assert raw_audio.ndim == 1
    assert audio.SAMPLE_RATE * 10 < raw_audio.shape[0] < audio.SAMPLE_RATE * 12
    assert 0 < raw_audio.std() < 1

    mel_from_audio = audio.log_mel_spectrogram(raw_audio)
    mel_from_file = audio.log_mel_spectrogram(audio_path)

    assert numpy.allclose(mel_from_audio, mel_from_file)
    assert mel_from_audio.max() - mel_from_audio.min() <= 2.0
