import os
import streamlit as st
import cv2
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from fpdf import FPDF
import sounddevice as sd
import queue
import time
import requests

# Streamlit UI Configuration
st.set_page_config(page_title="Mental Health Diagnostic Tool", layout="wide")

# Paths for Models
FACIAL_MODEL_PATH = "./model/emotion_recognition_model.h5"
SPEECH_MODEL_PATH = "./model/depression_audio_model.h5"

# Load Emotion Recognition Model
if not os.path.exists(FACIAL_MODEL_PATH):
    st.error("❌ Facial Emotion Model Not Found! Check the path.")
    st.stop()
facial_model = load_model(FACIAL_MODEL_PATH)

# Load Speech-Based Depression Model
if not os.path.exists(SPEECH_MODEL_PATH):
    st.error("❌ Depression Audio Model Not Found! Check the path.")
    st.stop()
speech_model = load_model(SPEECH_MODEL_PATH)

st.success("✅ All Models Loaded Successfully!")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to Process Images for Emotion Detection
def preprocess_image(image):
    img = np.array(image.convert("L"))  
    img_resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0
    img_final = np.expand_dims(img_normalized, axis=0)
    img_final = np.expand_dims(img_final, axis=-1)
    return img_final

# Function for Speech Feature Extraction
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    pitch_mean = np.mean(librosa.piptrack(y=y, sr=sr))
    pitch_std = np.std(librosa.piptrack(y=y, sr=sr))
    energy_mean = np.mean(librosa.feature.rms(y=y))
    energy_std = np.std(librosa.feature.rms(y=y))

    features = np.array([mfcc_mean.tolist() + mfcc_std.tolist() + 
                          [pitch_mean, pitch_std, energy_mean, energy_std]])
    
    return np.expand_dims(features, axis=1)

# Function for Real-Time Audio Recording
q = queue.Queue()
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def process_real_time_audio():
    st.subheader("🎙️ Speak Now... AI is Listening")
    duration = 5
    sr = 16000
    st.write("⏳ Listening...")
    with sd.InputStream(callback=callback, samplerate=sr, channels=1):
        time.sleep(duration)

    audio_data = []
    while not q.empty():
        audio_data.extend(q.get())

    audio_data = np.array(audio_data).flatten()
    audio_features = extract_audio_features(audio_data)
    prediction = speech_model.predict(audio_features)
    depression_label = "Depressed 😔" if prediction[0][0] > 0.5 else "Not Depressed 😊"

    st.write(f"🎭 **Live Prediction:** {depression_label} ({prediction[0][0]:.2f})")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a feature", ["Live Video", "Upload Image", "Upload Audio", "Live Speech", "Chat"])

# 1. **Live Facial Emotion Detection**
if app_mode == "Live Video":
    st.subheader("📹 Live Facial Emotion Detection")
    run = st.checkbox("Start Camera")
    frame_window = st.image([])

    if run:
        camera = cv2.VideoCapture(0)
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("❌ Failed to capture video!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
            normalized_img = resized_img / 255.0
            input_img = np.expand_dims(normalized_img, axis=(0, -1))

            prediction = facial_model.predict(input_img)
            emotion_detected = emotion_labels[np.argmax(prediction)]

            cv2.putText(frame, f"Emotion: {emotion_detected}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_window.image(frame, channels="BGR")

        camera.release()

# 2. **Upload Image for Emotion Detection**
elif app_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image for Emotion Detection", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)
        processed_img = preprocess_image(image)
        prediction = facial_model.predict(processed_img)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        st.write(f"🎭 **Predicted Emotion:** {predicted_emotion}")

# 3. **Upload Audio for Depression Detection**
elif app_mode == "Upload Audio":
    uploaded_audio = st.file_uploader("Upload an Audio File", type=["wav", "mp3", "ogg"])
    
    if uploaded_audio:
        audio_path = f"./temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())

        st.audio(audio_path, format="audio/wav")
        audio_features = extract_audio_features(audio_path)
        prediction = speech_model.predict(audio_features)
        depression_label = "Depressed 😔" if prediction[0][0] > 0.5 else "Not Depressed 😊"
        st.write(f"🧠 **Prediction:** {depression_label} ({prediction[0][0]:.2f})")

# 4. **Live Speech-Based Depression Detection**
elif app_mode == "Live Speech":
    process_real_time_audio()

# 5. **AI Chatbot for Mental Health Support**
elif app_mode == "Chat":
   


    # Define the FastAPI endpoint
    API_URL = "http://127.0.0.1:8000/mental-health-assessment"

    st.subheader("💬 Chat with the AI")

    # Get user input
    name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)
    user_response = st.text_area("How are you feeling today?")

    # Submit button
    if st.button("Submit"):
        if name and age and user_response:  # Ensure all fields are filled
            # Create JSON payload
            payload = {
                "name": name,
                "age": age,
                "responses": {"message": user_response}
            }
            
            try:
                # Send POST request to FastAPI
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    st.write(f"🤖 AI: {data.get("assessment")}")
                    
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
        else:
            st.warning("Please fill in all fields.")

