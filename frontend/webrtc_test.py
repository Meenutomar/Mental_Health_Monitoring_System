import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

webrtc_ctx = webrtc_streamer(
    key="audio-stream",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
)


if webrtc_ctx:
    st.write("🛠 WebRTC Status:", webrtc_ctx.state)
    st.write("📡 ICE Connection State:", getattr(webrtc_ctx, "iceConnectionState", "Not Available"))
    st.write("📡 ICE Gathering State:", getattr(webrtc_ctx, "iceGatheringState", "Not Available"))

if st.button("Restart WebRTC"):
    webrtc_ctx.stop()
    webrtc_ctx.start()
