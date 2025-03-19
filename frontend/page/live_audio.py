import streamlit as st
import websocket
import sounddevice as sd
import queue
import threading
import numpy as np
import simpleaudio as sa

API_AUDIO_URL = "ws://localhost:8000/audiostream"
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    q.put(indata.tobytes())

def start_audio_stream():
    ws = websocket.create_connection(API_AUDIO_URL)
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=audio_callback):
        while st.session_state.audio_active:
            audio_data = q.get()
            if audio_data:
                ws.send(audio_data)  # Send audio to backend
                ai_audio = ws.recv()  # Receive AI response
                play_audio(ai_audio)
    ws.close()

def play_audio(audio_data):
    wave_obj = sa.WaveObject(audio_data, num_channels=1, bytes_per_sample=2, sample_rate=16000)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def run():
    st.subheader("🎙️ Live Audio")
    
    if "audio_active" not in st.session_state:
        st.session_state.audio_active = False
    
    if st.button("🎤 Start Conversation"):
        st.session_state.audio_active = True
        threading.Thread(target=start_audio_stream, daemon=True).start()
        st.rerun()
    
    if st.session_state.audio_active:
        if st.button("🛑 Stop Conversation"):
            st.session_state.audio_active = False
            st.rerun()
