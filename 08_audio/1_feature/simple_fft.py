import numpy
from matplotlib import pyplot

sample_rate = 1024
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
