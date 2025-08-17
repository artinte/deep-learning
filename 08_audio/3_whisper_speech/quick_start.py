import os
import whisperspeech
import torchaudio
from whisperspeech.pipeline import Pipeline

pipe = Pipeline(
    t2s_ref="whisperspeech/whisperspeech:t2s-v1.95-small-8lang.model",
    s2a_ref="whisperspeech/whisperspeech:s2a-v1.95-medium-7lang.model",
)

audio_tensor = pipe.generate(
    "This is the first demo of Whisper Speech, a fully open source text-to-speech model trained by Collabora and Lion on the Juwels supercomputer."
)

output_dir = "temp"
os.makedirs(output_dir, exist_ok=True)
torchaudio.save(os.path.join(output_dir, 'temp.mp3'), audio_tensor.cpu(), 24000)
