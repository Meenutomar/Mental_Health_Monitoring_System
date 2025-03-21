import sounddevice as sd

def test_audio():
    print("🎤 Testing microphone...")
    duration = 5  # Seconds
    fs = 44100  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait for recording to finish
    print("✅ Audio recorded!", recording)

test_audio()
