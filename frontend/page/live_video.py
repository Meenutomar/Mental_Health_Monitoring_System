import streamlit as st
import requests
import base64
import numpy as np
import threading
import websocket
import json
import sounddevice as sd
import queue
import cv2

API_VIDEO_URL = "ws://localhost:8000/videostream"

st.title("🧠 AI Mental Health Live Conversation")

# ✅ Properly initialize session state variables at the very beginning
if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_started" not in st.session_state:
    st.session_state.video_started = False  # Ensure it's initialized before use

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    q.put(bytes(indata))

def start_audio_stream():
    ws = websocket.create_connection(API_VIDEO_URL)
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while st.session_state.video_started:
            audio_data = q.get()
            ws.send(audio_data)
            response = ws.recv()
            st.session_state.messages.append({"sender": "AI", "text": response})
            st.rerun()

def capture_video_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")
    return None

def send_video_frame():
    ws = websocket.create_connection(API_VIDEO_URL)
    while st.session_state.video_started:
        frame_data = capture_video_frame()
        if frame_data:
            ws.send(json.dumps({"type": "video", "frame": frame_data}))
    ws.close()

def run():
    st.subheader("🎥 Live Video")

    # ✅ Start Button
    if st.button("▶️ Start Video"):
        st.session_state.video_started = True
        threading.Thread(target=start_audio_stream, daemon=True).start()
        threading.Thread(target=send_video_frame, daemon=True).start()
        st.rerun()

    # ✅ Stop Button (Only visible when video is running)
    if st.session_state.video_started:
        if st.button("🛑 Stop Video"):
            st.session_state.video_started = False
            st.rerun()  # Refresh UI to hide the video feed

    if st.session_state.video_started:
        st.camera_input("Webcam Feed")  # Display webcam feed

    st.subheader("💬 AI Mental Health Chat")
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            if msg["sender"] == "User":
                st.markdown(f"**You:** {msg['text']}")
            else:
                st.markdown(f"**AI:** {msg['text']}")

    user_input = st.text_input("Type your message:")
    if st.button("Send"):
        if user_input:
            st.session_state.messages.append({"sender": "User", "text": user_input})
            ws = websocket.create_connection(API_VIDEO_URL)
            ws.send(json.dumps({"type": "text", "message": user_input}))
            response = ws.recv()
            st.session_state.messages.append({"sender": "AI", "text": response})
            ws.close()
            st.rerun()
