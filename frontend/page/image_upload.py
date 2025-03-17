import streamlit as st
import requests
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Use API URL from .env
API_URL = os.getenv("API_URL")

def run():
    st.subheader("Upload Image for Emotion Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)

        API_URL = "http://127.0.0.1:8000/predict-emotion/"
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            st.write(f"🎭 **Predicted Emotion:** {response.json()['emotion']}")
        else:
            st.error("Error: Unable to process image.")
