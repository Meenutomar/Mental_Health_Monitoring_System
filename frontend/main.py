import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page layout
st.set_page_config(page_title="Mental Health Diagnostic Tool", layout="wide")

# Custom CSS for fixed header and footer
st.markdown(
    """
    <style>
    /* Fixed Header */
    .header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #0e76a8;
        color: white;
        text-align: center;
        padding: 15px;
        font-size: 24px;
        font-weight: bold;
        z-index: 1000;
    }

    /* Fixed Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0e76a8;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 1000;
    }

    /* Adjust content to avoid overlap */
    .content {
        margin-top: 60px; /* Space for the fixed header */
        margin-bottom: 50px; /* Space for the fixed footer */
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Inject fixed header
st.markdown('<div class="header">Mental Health Diagnostic Tool</div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a feature", ["Video Feed", "Upload Image", "Mood Form", "Chat", "Report"])

# 1. Video Feed (Using OpenCV)
if app_mode == "Video Feed":
    st.subheader("Live Camera Feed")

    # Create two columns (one for the checkbox, one for the video feed)
    col1, col2 = st.columns([1, 3])  # Adjust width ratio as needed

    with col1:
        run = st.checkbox("Start Camera")

    with col2:
        frame_window = st.empty()  # Placeholder for video

    if run:
        camera = cv2.VideoCapture(0)
        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame, caption="Live Feed", use_container_width=True)
        camera.release()

# 2. Image Upload
elif app_mode == "Upload Image":
    st.subheader("Upload an Image for Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.success("Image successfully uploaded!")

# 3. Mood Form
elif app_mode == "Mood Form":
    st.subheader("Mood Assessment Form")
    
    with st.form(key="mood_form"):
        mood = st.selectbox("How are you feeling today?", ["Happy", "Anxious", "Stressed", "Depressed", "Neutral"])
        sleep = st.selectbox("How was your sleep?", ["Good", "Average", "Poor"])
        appetite = st.selectbox("How is your appetite?", ["Normal", "Increased", "Decreased"])
        energy = st.selectbox("Energy levels today?", ["High", "Moderate", "Low"])
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            st.success("Mood assessment recorded!")

# 4. Free Text Chat
elif app_mode == "Chat":
    st.subheader("Chat with the AI")
    user_input = st.text_area("Express your thoughts here:")
    
    if st.button("Submit"):
        st.write(f"You said: {user_input}")
        st.success("Message recorded!")

# 5. Generate Report
elif app_mode == "Report":
    st.subheader("Mental Health Report")
    
    st.write("Mood: Happy")
    st.write("Sleep: Good")
    st.write("Appetite: Normal")
    st.write("Energy Levels: High")
    st.write("Your thoughts: 'I am feeling great today!'")

    st.markdown("### **Summary:** Based on your recent entries, your mental health seems stable. Keep maintaining a balanced routine.")

# End content wrapper
st.markdown('</div>', unsafe_allow_html=True)

# Inject fixed footer
st.markdown('<div class="footer">© 2025 Mental Health AI | All Rights Reserved</div>', unsafe_allow_html=True)