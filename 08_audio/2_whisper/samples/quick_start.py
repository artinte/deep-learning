import whisper

model = whisper.load_model("small")
result = model.transcribe('res/audio/voice_sample.mp3')
print(result['text'])
