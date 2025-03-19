from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
import whisper
import asyncio
import numpy as np
import edge_tts
import tempfile
from chatbot import mental_health_assessment  # ✅ Import chatbot logic

app = FastAPI()

# Load Whisper Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

async def speech_to_text(audio_data: bytes) -> str:
    """Convert speech audio to text using Whisper."""
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    result = model.transcribe(audio_np)
    return result["text"].strip()

async def text_to_speech(text: str) -> bytes:
    """Convert AI-generated text to speech audio using edge-tts."""
    tts = edge_tts.Communicate(text, "en-US-JennyNeural")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
        await tts.save(temp_audio.name)
        temp_audio.seek(0)
        return temp_audio.read()

@app.websocket("/audiostream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("I am listening")
    try:
        while True:
            audio_data = await websocket.receive_bytes()  # Receive user audio
            user_text = await speech_to_text(audio_data)
            print("User:", user_text)

            ai_response = await mental_health_assessment({"name": "User", "age": 25, "message": user_text})
            ai_text = ai_response["assessment"]
            print("AI:", ai_text)

            ai_audio = await text_to_speech(ai_text)
            await websocket.send_bytes(ai_audio)  # Send audio response to frontend
    except WebSocketDisconnect:
        print("WebSocket disconnected")
