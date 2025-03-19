import streamlit as st
import websocket
import json
import sounddevice as sd
import numpy as np
import queue
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import threading

# Initialize session state
if "audio_started" not in st.session_state:
    st.session_state.audio_started = False
if "messages" not in st.session_state:
    st.session_state.messages = []

API_AUDIO_URL = "ws://localhost:8000/audiostream"
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    q.put(indata.tobytes())

def start_audio_stream():
    ws = websocket.create_connection(API_AUDIO_URL)
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while st.session_state.audio_started:
            audio_data = q.get()
            if audio_data:
                ws.send(audio_data)
                response = ws.recv()
                play_audio(response)  # Play AI's response as audio
                st.session_state.messages.append({"sender": "AI", "text": response})
                st.rerun()

def play_audio(audio_data):
    """Play received AI response audio."""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_data)
        temp_audio.flush()
        audio = AudioSegment.from_file(temp_audio.name, format="mp3")
        play(audio)

def run():
    st.subheader("🎙️ AI Mental Health Audio Chat")

    if st.button("🎧 Start Chat"):
        st.session_state.audio_started = True
        threading.Thread(target=start_audio_stream, daemon=True).start()
        st.rerun()

    if st.session_state.audio_started:
        if st.button("🛑 Stop Chat"):
            st.session_state.audio_started = False
            st.rerun()

    st.subheader("💬 Chat Log")
    for msg in st.session_state.messages:
        st.markdown(f"**{msg['sender']}:** {msg['text']}")

if __name__ == "__main__":
    run()
