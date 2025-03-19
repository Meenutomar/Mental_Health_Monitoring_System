import streamlit as st
import websocket
import base64

API_AUDIO_URL = "ws://localhost:8000/audiostream"

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

def run():
    """Main Streamlit UI."""
    st.subheader("🎙️ Lets Talk")

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


