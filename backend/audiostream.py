from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import edge_tts
import asyncio

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
