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
    await websocket.send_text("Connection established")  # Test response
    # ✅ Greet user with an initial voice message
    welcome_message = "Hello! How are you feeling today?"
    welcome_audio = await text_to_speech(welcome_message)
    await websocket.send_bytes(welcome_audio)

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            result = model.transcribe(audio_np)
            user_response = result["text"].strip()
            print("User:", user_response)

            # Call chatbot logic
            ai_response = await mental_health_assessment({"name": "User", "age": 25, "message": user_response})

            # Convert AI response to speech
            ai_audio = await text_to_speech(ai_response["assessment"])
            await websocket.send_bytes(ai_audio)

    except WebSocketDisconnect:
        print("WebSocket disconnected")

async def text_to_speech(text):
    """Convert text to speech using edge-tts."""
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes
