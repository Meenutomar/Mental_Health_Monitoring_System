from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import edge_tts
import asyncio
import speech_recognition as sr
from io import BytesIO
import websockets
import json

router = APIRouter()

@router.websocket("/audiostream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connection established.")

    # ✅ Send welcome message (audio)
    welcome_message = "Hello! How are you feeling today?"
    welcome_audio = await text_to_speech(welcome_message)
    await websocket.send_bytes(welcome_audio)  # Send as audio

    print("✅ Welcome message sent. Closing connection.")
    await websocket.close()  # Close WebSocket after sending welcome message

async def text_to_speech(text):
    """Convert text to speech using edge-tts."""
    print(f"🔊 Converting text to speech: {text}")
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    print(f"✅ Generated {len(audio_bytes)} bytes of audio.")
    return audio_bytes

@router.websocket("/transcribe")
async def websocket_transcription(websocket: WebSocket):
    """WebSocket to receive user audio and return transcribed text."""
    print('Establishing Transcribe WebSocket Connection...')
    await websocket.accept()
    print("✅ Transcription WebSocket connection established.")

    try:
        audio_bytes = await websocket.receive_bytes()  # Receive audio from frontend
        print(f"📥 Received {len(audio_bytes)} bytes of user audio.")

        # ✅ Convert speech to text
        transcribed_text = await speech_to_text(audio_bytes)
        print(f"📝 Transcribed text: {transcribed_text}")

        # ✅ Respond to the user
        response_text = f"You said: {transcribed_text}. How can I assist you further?"
        response_audio = await text_to_speech(response_text)

        # ✅ Send transcribed text and response back to frontend
        response_payload = {
            "text": transcribed_text,
            "audio": response_audio.decode("latin1")  # Convert bytes to string (base64-like encoding)
        }
        await websocket.send_text(json.dumps(response_payload))

    except Exception as e:
        print(f"⚠️ Error processing audio: {e}")
        await websocket.send_text(json.dumps({"error": str(e)}))

    finally:
        print("✅ Closing transcription WebSocket connection.")
        await websocket.close()

async def speech_to_text(audio_bytes):
    """Convert speech audio bytes to text using SpeechRecognition."""
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as audio_file:
            audio_data = recognizer.record(audio_file)
            text = recognizer.recognize_google(audio_data)  # Using Google's STT
            return text
    except sr.UnknownValueError:
        return "I couldn't understand that."
    except sr.RequestError:
        return "Speech-to-text service is unavailable."
