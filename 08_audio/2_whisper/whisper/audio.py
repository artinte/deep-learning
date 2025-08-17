import numpy
import subprocess
import torch
import utils


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
# 480000 samples in a 30-second chunk
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
# 3000 frames in a mel spectrogram input
N_FRAMES = utils.exact_div(N_SAMPLES, HOP_LENGTH)
# the initial convolutions has stride 2
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
# 10ms per audio frame
FRAMES_PER_SECOND = utils.exact_div(SAMPLE_RATE, HOP_LENGTH)
# 20ms per audio token
TOKENS_PER_SECOND = utils.exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)

def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    
    return numpy.frombuffer(out, numpy.int16).flatten().astype(numpy.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int=-1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = torch.nn.functional.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)
        
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = numpy.pad(array, pad_widths)
    
    return array

