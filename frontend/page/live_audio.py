import streamlit as st
import websocket
import base64
import asyncio
import json
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

API_AUDIO_URL = "ws://localhost:8000/audiostream"
API_TRANSCRIBE_URL = "http://localhost:8000/transcribe"

# ✅ Convert Image to Base64
def img_to_bytes(img_path):
    """Converts an image file to base64 encoded bytes."""
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# ✅ Load RoboMH Icon (Ensure correct path)
image_path = "./assets/logo.png"
image_bytes = img_to_bytes(image_path)

def update_logs(new_log):
    """Dynamically updates logs like a console."""
    if "logs" not in st.session_state:
        st.session_state.logs = []
    
    st.session_state.logs.append(new_log)
    print(new_log)

    # ✅ Update logs in real-time **only after log_placeholder is defined**
    if "log_placeholder" in st.session_state:
        st.session_state.log_placeholder.code("\n".join(st.session_state.logs), language="bash")

def get_welcome_audio():
    """Fetch welcome message from backend, play it, and log steps."""
    try:
        update_logs("🎙️ Connecting to WebSocket for welcome message...")
        ws = websocket.create_connection(API_AUDIO_URL)

        # ✅ Receive the audio bytes from WebSocket
        welcome_audio = ws.recv()
        if isinstance(welcome_audio, bytes):
            update_logs(f"✅ Received {len(welcome_audio)} bytes of welcome audio")
            st.session_state.is_speaking = True  # 🔥 Mark speaking state
            update_logs("🤖 RoboMH is speaking...")
            auto_play_audio(welcome_audio)  # 🔥 Auto-play function
        
        ws.close()
        update_logs("✅ WebSocket closed after receiving welcome message.")
    except Exception as e:
        update_logs(f"⚠️ Error fetching welcome message: {e}")

def auto_play_audio(audio_bytes):
    """Automatically play audio and log the process."""
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    update_logs("🔊 Playing welcome audio...")
    
    audio_html = f"""
        <audio id="audio-player" autoplay>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        <script>
            var audio = document.getElementById("audio-player");

            audio.onended = function() {{
                fetch('/_stcore_update', {{ method: 'POST' }});  // ✅ Force UI refresh
            }};
        </script>
    """

    st.markdown(audio_html, unsafe_allow_html=True)
    update_logs("🎧 Audio playback started.")

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []
        update_logs("🛠️ AudioProcessor initialized ✅")

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        update_logs("🎤 Audio frame received ✅")
        print("🎤 Audio frame received ✅")  # Debugging

        self.frames.append(frame)
        return frame  # ✅ Return frame properly

    def get_audio_data(self):
        """Convert captured audio frames into WAV bytes."""
        if not self.frames:
            update_logs("⚠️ No audio frames captured! 🚨")
            return b""  # Return empty bytes if no frames

        update_logs(f"🎙️ Capturing {len(self.frames)} audio frames...")
        audio_bytes = b"".join([frame.to_ndarray().tobytes() for frame in self.frames])

        update_logs(f"🔊 Generated {len(audio_bytes)} bytes of audio data")
        return audio_bytes

def send_audio_to_backend(audio_bytes):
    """Send recorded audio to backend for transcription."""
    try:
        update_logs("📤 Sending user audio for transcription...")
        print('📤 Sending user audio for transcription...')
        ws = websocket.create_connection(API_TRANSCRIBE_URL)

        ws.send(audio_bytes)
        response = ws.recv()
        ws.close()

        # ✅ Handle backend response
        if response:
            response_data = json.loads(response)
            transcribed_text = response_data.get("text", "")
            update_logs(f"🗣️ User: {transcribed_text}")
            return transcribed_text
        else:
            update_logs("⚠️ Error: No response received from backend.")
            return None
    except Exception as e:
        update_logs(f"⚠️ Error during transcription: {e}")
        return None

def run():
    """Main Streamlit UI."""
    st.title("🎙️ AI Mental Health Chat")

    # ✅ Initialize session state at the START of run()
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "is_speaking" not in st.session_state:
        st.session_state.is_speaking = False

    # ✅ Button to Start Chat
    if st.button("🎧 Start Chat"):
        update_logs("🟢 Start button clicked")
        get_welcome_audio()  # 🔥 Fetch & play welcome message again

    # ✅ Define Logs Section **Below Start Chat Button**
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 20px;">
            <img src="data:image/png;base64,{image_bytes}" width="40" height="40">
            <h5 style="margin: 0;">Logs</h5>
        </div>
        <hr style="border: 1px solid gray;">
        """,
        unsafe_allow_html=True,
    )

    # ✅ Define Placeholder for Logs (AFTER Button)
    if "log_placeholder" not in st.session_state:
        st.session_state.log_placeholder = st.empty()  # ✅ Now logs appear **below** the button

    # ✅ Render initial logs
    st.session_state.log_placeholder.code("\n".join(st.session_state.logs), language="bash")

    # ✅ Live Audio Capture (User Response)
    st.subheader("🎙️ Speak Now")
    webrtc_ctx = webrtc_streamer(
        key="user-audio",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={
            "video": True,
            "audio": True  # 🔥 Disable echo cancellation
        },
    )

    if webrtc_ctx.audio_processor is None:
        update_logs("⚠️ Audio processor is not initialized!")
    else:
         update_logs("🟢 Audio processor is  initialized!")

    # ✅ Capture and Send Audio on Button Click
    if webrtc_ctx.audio_processor and st.button("🗣️ Send Response"):
        update_logs("🟢 Send Response Clicked")
        audio_data = webrtc_ctx.audio_processor.get_audio_data()
        update_logs("🟢 Fetched Audio Data")
        if audio_data:
            update_logs("🟢 Sending audio to backend")
            transcribed_text = send_audio_to_backend(audio_data)
            update_logs("🟢 Received Response")
            if transcribed_text:
                st.session_state.logs.append(f"🗣️ {transcribed_text}")  # ✅ Add transcript to logs
                st.session_state.log_placeholder.code("\n".join(st.session_state.logs), language="bash")

# ✅ Run the app
if __name__ == "__main__":
    run()
