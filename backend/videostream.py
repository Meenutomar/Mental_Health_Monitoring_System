from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import torch
import whisper
import numpy as np
import base64
from deepface import DeepFace
from chatbot import mental_health_assessment  # ✅ Import function from chat.py

router = APIRouter()

# Load Whisper Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

@router.websocket("/videostream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_bytes()  # Receive audio data from frontend
            
            # Convert audio bytes to NumPy array
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe speech using Whisper
            result = model.transcribe(audio_np)
            user_response = result["text"].strip()
            print("User:", user_response)

            # Facial Emotion Analysis
            face_emotion = analyze_facial_expression()
            print("Detected Emotion:", face_emotion)

            # ✅ Call `mental_health_assessment()` from chat.py
            ai_question = await mental_health_assessment({"name": "User", "age": 25, "message": user_response})  # Sample age
            await websocket.send_text(ai_question["assessment"])  # Send AI response to frontend

    except WebSocketDisconnect:
        print("WebSocket disconnected")

def analyze_facial_expression():
    """Analyze emotions from video frames using DeepFace."""
    frame = get_current_video_frame()
    if frame is not None:
        analysis = DeepFace.analyze(frame, actions=['emotion'])
        return analysis[0]['dominant_emotion']
    return "neutral"

def get_current_video_frame():
    """Retrieve and decode the latest video frame."""
    frame_data = st.session_state.get("latest_video_frame", None)
    if frame_data:
        img_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return None
