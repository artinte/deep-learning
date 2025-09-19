import numpy
from matplotlib import pyplot

sample_rate = 2048
duration = 1
t = numpy.linspace(0, duration, sample_rate, endpoint=False)

freq1 = 50
freq2 = 200
signal = numpy.sin(2 * numpy.pi * freq1 * t) + 0.5 * numpy.sin(2 * numpy.pi * freq2 * t)

fft_result = numpy.fft.fft(signal)
freqs = numpy.fft.fftfreq(len(t), 1 / sample_rate)

positive_freqs = freqs[:sample_rate // 2]
magnitude = numpy.abs(fft_result)[:sample_rate // 2]

pyplot.subplot(1, 2, 1)
pyplot.plot(t, signal)
pyplot.title("Time Domain (Signal)")
pyplot.xlabel("Time (s)")
pyplot.ylabel("Amplitude")

pyplot.subplot(1, 2, 2)
pyplot.plot(positive_freqs, magnitude)
pyplot.title("Frequency Domain (FFT)")
pyplot.xlabel("Frequency (Hz)")
pyplot.ylabel("Magnitude")

pyplot.tight_layout()
pyplot.show()


def frame_signal(signal,
                 sample_rate,
                 frame_size=0.025,
                 frame_stride=0.01):
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(signal)
    
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step)) + 1
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(signal, z)
    
    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + \
              numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    
    return frames


def apply_window(frames, window_type='hann'):
    if window_type == 'hann':
        window = numpy.hanning(frames.shape[1])
    elif window_type == 'hamming':
        window = numpy.hamming(frames.shape[1])
    else:
        window = numpy.ones(frames.shape[1])
    
    return frames * window


def compute_fft(frames, n_fft=512):
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * ((mag_frames) **2))
    
    return pow_frames


def hz_to_mel(hz):
    return 2595 * numpy.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10** (mel / 2595) - 1)


def create_mel_filters(sample_rate, n_fft, n_mels=40):
    max_freq = sample_rate // 2
    
    mel_points = numpy.linspace(hz_to_mel(0), hz_to_mel(max_freq), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    bin = numpy.floor((n_fft + 1) * hz_points / sample_rate)
    
    fbank = numpy.zeros((n_mels, int(numpy.floor(n_fft / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        
        if f_m_minus != f_m:
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])

        if f_m != f_m_plus:
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    return fbank

def compute_mel_spectrogram(signal,
                            sample_rate,
                            frame_size=0.025,
                            frame_stride=0.01,
                            n_fft=512,
                            n_mels=40):
    
    frames = frame_signal(signal, sample_rate, frame_size, frame_stride)
    
    windowed_frames = apply_window(frames)
    
    pow_spec = compute_fft(windowed_frames, n_fft)
    
    mel_filters = create_mel_filters(sample_rate, n_fft, n_mels)
    
    mel_spectrogram = numpy.dot(pow_spec, mel_filters.T)

    mel_spectrogram = 20 * numpy.log10(mel_spectrogram + 1e-10)

    return mel_spectrogram

sample_rate = 16000
duration = 1.0
t = numpy.linspace(0, duration, int(sample_rate * duration), endpoint=False)

signal = 0.5 * numpy.sin(2 * numpy.pi * 440 * t)  # 440Hz (A4)
signal += 0.3 * numpy.sin(2 * numpy.pi * 880 * t) # 880Hz (A5)
signal += 0.2 * numpy.sin(2 * numpy.pi * 1320 * t) # 1320Hz (E6)

mel_spec = compute_mel_spectrogram(
    signal,
    sample_rate,
    # window size 25ms
    frame_size=0.025,
    # step size 10ms
    frame_stride=0.01,
    n_fft=512,
    n_mels=40
)

pyplot.imshow(mel_spec.T, origin='lower', aspect='auto', 
            extent=[0, duration, 0, sample_rate//2])
pyplot.title('Mel Spectrogram')
pyplot.xlabel('Time (s)')
pyplot.ylabel('Mel Frequency')
pyplot.colorbar(label='Intensity (dB)')
pyplot.tight_layout()
pyplot.show()


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=512,
    win_length=400,  # 25ms
    hop_length=160,  # 10ms
    n_mels=40
)

mel_spec = mel_transform(signal)
mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

pyplot.imshow(mel_spec_db[0].numpy(), origin='lower', aspect='auto')
pyplot.title('Mel Spectrogram')
pyplot.xlabel('Time')
pyplot.ylabel('Mel Bins')
pyplot.colorbar()
pyplot.show()
