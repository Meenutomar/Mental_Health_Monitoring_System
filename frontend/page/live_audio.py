import streamlit as st
import websocket
import sounddevice as sd
import numpy as np
import queue
import threading
import time

# ✅ Global flag for audio state (Thread-Safe)
audio_started_flag = False

# WebSocket API URL
API_AUDIO_URL = "ws://localhost:8000/audiostream"

# Queue for storing audio chunks
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Captures microphone input and adds it to the queue."""
    print("Inside audio callback:", status)
    data =  indata.copy()
    if status:
        print(status)
    audio_queue.put(data)

def start_audio_stream():
    """Manages the WebSocket connection and streams audio."""
    global audio_started_flag  # ✅ Use global flag instead of session state
    try:
        print("🛠️ Starting WebSocket Thread...")

        ws = websocket.create_connection(API_AUDIO_URL)
        print("✅ WebSocket connection established!")

        # 🔥 Start audio recording  
        stream = sd.InputStream(callback=audio_callback, samplerate=16000, channels=1, dtype="int16")
        stream.start()  # ✅ Start capturing audio
        print("🎤 Microphone stream started!")

        while audio_started_flag:  # ✅ Using thread-safe flag instead of session state
            print("🔄 Loop running inside WebSocket thread!")

            if not audio_queue.empty():
                audio_data = audio_queue.get()
                print(f"🎙️ Sending {len(audio_data)} bytes of audio...")
                ws.send(audio_data.tobytes())

                response = ws.recv()
                print(f"🤖 Received AI response: {type(response)}")

                if isinstance(response, bytes):  # AI response as audio
                    st.session_state.audio_bytes = response
                else:  # AI response as text
                    st.session_state.messages.append({"sender": "AI", "text": response})

                time.sleep(0.1)  # Prevents high CPU usage
            else:
                print("⚠️ No audio data in queue!")

        print("❌ Stopping microphone stream and closing WebSocket")
        stream.stop()
        stream.close()
        ws.close()

    except Exception as e:
        print(f"⚠️ WebSocket Error: {e}")

def run():
    """Streamlit UI for AI Mental Health Chat."""
    global audio_started_flag  # ✅ Use global variable for thread safety

    st.subheader("🎙️ Live Audio")

    if st.button("🎧 Start Chat"):
        print("🟢 Start button clicked")
        
        if not audio_started_flag:
            audio_started_flag = True  # ✅ Set global flag

            print("✅ audio_started set to True")

            # 🔥 Start WebSocket communication in a separate thread
            thread = threading.Thread(target=start_audio_stream, daemon=True)
            thread.start()

            time.sleep(1)  # Give some time to start the WebSocket

            if thread.is_alive():
                print("🚀 WebSocket thread started successfully!")
            else:
                print("❌ WebSocket thread failed to start!")

            st.rerun()  # ✅ Refresh UI to reflect session state changes

    if audio_started_flag:
        if st.button("🛑 Stop Chat"):
            audio_started_flag = False  # ✅ Stop the thread safely
            st.rerun()  # ✅ Refresh UI after stopping

    st.subheader("💬 Chat Log")
    for msg in st.session_state.get("messages", []):
        st.markdown(f"**{msg['sender']}:** {msg['text']}")

    # ✅ Play AI response audio
    if st.session_state.get("audio_bytes"):
        st.audio(st.session_state.audio_bytes, format="audio/wav")

if __name__ == "__main__":
    run()
