import streamlit as st
import requests
import time

st.title("🎤 AI Mental Health Assessment - Voice Recording")

# JavaScript for Audio Recording (Now using Streamlit Button Events)
js_code = """
    <script>
    let mediaRecorder;
    let audioChunks = [];

    function startRecording() {
        console.log("🎙 Starting Recording...");  // Debug
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                console.log("🔴 Recording Started...");
                audioChunks = [];

                mediaRecorder.addEventListener("dataavailable", event => {
                    audioChunks.push(event.data);
                });

                mediaRecorder.addEventListener("stop", () => {
                    console.log("⏹ Recording Stopped...");
                    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                    const reader = new FileReader();
                    
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        let base64AudioMessage = reader.result.split(',')[1];

                        console.log("📤 Sending Audio to Backend...");
                        fetch("http://127.0.0.1:8000/upload_audio", {  // Ensure URL is correct
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ audio_data: base64AudioMessage })
                        })
                        .then(response => response.json())
                        .then(data => console.log("✅ Response from Backend:", data))
                        .catch(error => console.error("❌ Fetch Error:", error));
                    };
                });
            })
            .catch(error => console.error("❌ Error Accessing Microphone:", error));
    }

    function stopRecording() {
        console.log("🛑 Stopping Recording...");
        mediaRecorder.stop();
    }
    </script>
"""

st.markdown(js_code, unsafe_allow_html=True)

if st.button("🎙 Start Recording"):
    st.write("Recording... Speak Now!")
    st.markdown('<script>startRecording();</script>', unsafe_allow_html=True)

if st.button("⏹ Stop Recording"):
    st.write("Processing audio...")
    st.markdown('<script>stopRecording();</script>', unsafe_allow_html=True)

# Wait for processing and fetch result from FastAPI
time.sleep(3)  # Simulating processing delay
response = requests.get("http://127.0.0.1:8000/get_result")  # Get AI analysis
if response.status_code == 200:
    st.write("✅ AI Analysis:", response.json()["message"])
