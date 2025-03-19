from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import torch
import whisper
import numpy as np
import edge_tts
import asyncio
from chatbot import mental_health_assessment

router = APIRouter()

# Load Whisper Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

@router.websocket("/audiostream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")  # Debugging
    
    # ✅ Send initial voice message
    welcome_message = "Hello! How are you feeling today?"
    welcome_audio = await text_to_speech(welcome_message)
    await websocket.send_bytes(welcome_audio)

    try:
        while True:
            print("Waiting for audio data...")  # Debugging
            data = await websocket.receive_bytes()
            
            if not data:
                print("Received empty audio data, skipping...")
                continue

            print(f"Received {len(data)} bytes of audio")  # Debugging

            # Convert bytes to numpy array for Whisper processing
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # ✅ Transcribe speech to text
            try:
                result = model.transcribe(audio_np)
                user_response = result["text"].strip()
                print("User:", user_response)
            except Exception as e:
                print(f"Whisper failed: {e}")
                continue  # Skip processing if transcription fails

            # ✅ Get AI response
            try:
                ai_response = await mental_health_assessment(
                    {"name": "User", "age": 25, "message": user_response}
                )
                ai_text = ai_response.get("assessment", "I couldn't understand. Please repeat.")
            except Exception as e:
                print(f"Chatbot error: {e}")
                ai_text = "Sorry, I encountered an issue."

            # ✅ Convert AI response to speech
            ai_audio = await text_to_speech(ai_text)
            await websocket.send_bytes(ai_audio)

    except WebSocketDisconnect:
        print("WebSocket disconnected")

async def text_to_speech(text):
    """Convert text to speech using edge-tts."""
    print(f"Converting text to speech: {text}")
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    print(f"Generated audio size: {len(audio_bytes)} bytes")  # Debugging
    return audio_bytes
