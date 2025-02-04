import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for, send_file
from fpdf import FPDF  # PDF generation

# ✅ Suppress TensorFlow Warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Initialize Flask app
app = Flask(__name__, static_folder="static")

# ✅ Define paths
MODEL_PATH = r"E:\SEM_4\MajorProject\model\emotion_recognition_model.h5"
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Load the trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ✅ Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 200.0
    img_final = np.expand_dims(img_normalized, axis=0)
    img_final = np.expand_dims(img_final, axis=-1)

    return img_final

# ✅ Function to convert responses to a score (0-100)
def convert_to_score(value):
    score_map = {
        "Rarely": 25, "Sometimes": 50, "Often": 75, "Always": 100,
        "No": 0, "Occasionally": 25, "Frequently": 75, "Every Night": 100,
        "Never": 0, "Few times a week": 50, "Daily": 100,
        "Very High": 100, "Moderate": 50, "Low": 25, "Very Low": 0
    }
    return score_map.get(value, 50)

# ✅ Function to generate structured report
def generate_report(emotion, responses):
    report = f"""
    <div class='report-container'>
        <h2>🧠 Mental Health & Well-being Report</h2>
        <hr>
        <h3>🎭 Detected Emotion: <span class='highlight'>{emotion}</span></h3>
        <hr>
    """

    questions = [
        ("🔹 Stress Level", "High stress detected! Consider relaxation techniques like deep breathing."),
        ("🌙 Sleep Quality", "Sleep disturbances detected. Try establishing a bedtime routine."),
        ("😰 Anxiety Symptoms", "High anxiety detected. Consider mindfulness exercises."),
        ("😊 Mood Analysis", "Low mood detected. Engage in enjoyable activities."),
        ("⚡ Energy & Fatigue", "Fatigue detected! Ensure proper hydration and nutrition."),
        ("🏃‍♂️ Physical Activity", "Lack of physical activity detected! Try simple daily exercises."),
        ("🤝 Social Life & Isolation", "Social isolation detected! Engage in social activities."),
        ("🌟 Self-Esteem", "Low self-esteem detected! Practice self-care and affirmations.")
    ]

    for i, (question, suggestion) in enumerate(questions):
        response = responses[i]  # Extract response from list
        score = convert_to_score(response)  # Convert response to score
        report += f"<div class='section'><h4>{question}</h4><p>{response} - {suggestion if score < 50 else '✅ You are doing well!'}</p></div>"

    report += """
        <div class='footer'>
            <p>📌 This report is for informational purposes only. Consult a professional if needed.</p>
        </div>
    </div>
    """
    return report

# ✅ Home route
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        processed_img = preprocess_image(file_path)
        if processed_img is None:
            return "Error: Invalid image!"

        prediction = model.predict(processed_img)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        # ✅ Collect responses
        responses = [
            request.form.get("stress_level"),
            request.form.get("sleep_issues"),
            request.form.get("anxiety_level"),
            request.form.get("mood"),
            request.form.get("fatigue"),
            request.form.get("exercise"),
            request.form.get("social_life"),
            request.form.get("self_esteem")
        ]

        # ✅ Generate the report
        report = generate_report(predicted_emotion, responses)

        return render_template("report.html", image=file.filename, emotion=predicted_emotion, report=report)

    return render_template("index.html")

# ✅ Generate PDF Report
@app.route("/download_pdf/<filename>/<emotion>")
def download_pdf(filename, emotion):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add Title
    pdf.cell(200, 10, "🧠 Mental Health & Emotion Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # Add Image
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(image_path):
        pdf.image(image_path, x=10, y=30, w=80)

    pdf.ln(75)  # Space after image

    # Add Emotion Analysis
    pdf.cell(200, 10, f"🎭 Detected Emotion: {emotion}", ln=True, align="C")
    pdf.ln(10)

    # Add Mental Health Assessment (Manually Format Text)
    pdf.multi_cell(0, 10, """
    🔍 Therapist Insights:
    - Stress Level
    - Sleep Issues
    - Anxiety Level
    - Mood
    - Fatigue
    - Physical Activity
    - Social Life
    - Self-Esteem
    """)

    # Add Disclaimer
    pdf.ln(10)
    pdf.cell(200, 10, "📌 This report is for informational purposes only. Consult a professional for advice.", ln=True, align="C")

    # Save the PDF
    pdf_path = f"static/uploads/{filename}.pdf"
    pdf.output(pdf_path)

    return send_file(pdf_path, as_attachment=True)

# ✅ Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
