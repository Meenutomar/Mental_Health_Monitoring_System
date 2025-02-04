import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from fpdf import FPDF

# Load the pre-trained emotion recognition model
MODEL_PATH = "E:/SEM_4/MajorProject/model/emotion_recognition_model.h5"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Please check the path.")
    st.stop()

model = load_model(MODEL_PATH)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Preprocess image function
def preprocess_image(image):
    img = np.array(image.convert("L"))  # Convert to grayscale
    img_resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0
    img_final = np.expand_dims(img_normalized, axis=0)
    img_final = np.expand_dims(img_final, axis=-1)
    return img_final

# Function to convert responses to score
def convert_to_score(value):
    score_map = {
        "Rarely": 25, "Sometimes": 50, "Often": 75, "Always": 100,
        "No": 0, "Occasionally": 25, "Frequently": 75, "Every Night": 100,
        "Never": 0, "Few times a week": 50, "Daily": 100,
        "Very High": 100, "Moderate": 50, "Low": 25, "Very Low": 0
    }
    return score_map.get(value, 50)

# Generate PDF report
def generate_pdf(filename, emotion, responses):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, "Mental Health & Emotion Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, f"Detected Emotion: {emotion}", ln=True, align="C")
    pdf.ln(10)
    
    questions = [
        "Stress Level", "Sleep Quality", "Anxiety Symptoms", "Mood Analysis", 
        "Energy & Fatigue", "Physical Activity", "Social Life & Isolation", "Self-Esteem"
    ]
    
    for q, r in zip(questions, responses):
        pdf.multi_cell(0, 10, f"{q}: {r}")
    
    pdf.ln(10)
    pdf.cell(200, 10, "This report is for informational purposes only.", ln=True, align="C")
    
    pdf_path = f"{filename}.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Streamlit UI
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
    # File uploader
    uploaded_file = st.file_uploader("Upload an Image for Emotion Detection", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)

        # Process Image
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        st.write(f"🎭 **Predicted Emotion:** {predicted_emotion}")

        # Mental Health Questions
        st.subheader("📝 Mental Health Questionnaire")
        
        responses = [
            st.radio("How often do you feel stressed?", ["Rarely", "Sometimes", "Often", "Always"]),
            st.radio("Do you experience sleep issues?", ["No", "Occasionally", "Frequently", "Every Night"]),
            st.radio("Do you experience anxiety symptoms?", ["Never", "Few times a week", "Daily"]),
            st.radio("How would you describe your mood?", ["Very Low", "Low", "Moderate", "Very High"]),
            st.radio("Do you feel fatigued or low in energy?", ["Never", "Sometimes", "Often", "Always"]),
            st.radio("How often do you engage in physical activity?", ["Never", "Few times a week", "Daily"]),
            st.radio("Do you feel socially isolated?", ["Never", "Sometimes", "Often", "Always"]),
            st.radio("How would you rate your self-esteem?", ["Very Low", "Low", "Moderate", "Very High"])
        ]

        if st.button("Generate Report"):
            st.session_state["emotion"] = predicted_emotion
            st.session_state["responses"] = responses
            pdf_path = generate_pdf("Mental_Health_Report", predicted_emotion, responses)
            
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(label="📥 Download Report", data=pdf_file, file_name="Mental_Health_Report.pdf", mime="application/pdf")


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

#st.title("🧠 Mental Health Diagnostic Tool")

