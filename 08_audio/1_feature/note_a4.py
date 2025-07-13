import numpy
from scipy.io.wavfile import write

sample_rate = 44100
duration = 2.0
frequency = 440

t = numpy.linspace(0, duration, int(sample_rate * duration), endpoint=False)
wave = 0.5 * numpy.sin(2 * numpy.pi * frequency * t)

max_amplitude = numpy.iinfo(numpy.int16).max
write('temp/sine_440hz.wav', sample_rate, (wave * max_amplitude).astype(numpy.int16))
