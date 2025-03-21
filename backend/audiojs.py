from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import base64
import io
import soundfile as sf

router = APIRouter()

class AudioData(BaseModel):
    audio_data: str  # Base64 encoded audio

@router.post("/upload_audio")
async def upload_audio(audio: AudioData):
    try:
        # Decode Base64 audio
        print('Inside Upload Audio')
        audio_bytes = base64.b64decode(audio.audio_data)
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        print('Inside Upload Audio 1')
        # Load audio and process it
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        print(f"✅ Received Audio - Sample Rate: {samplerate}, Duration: {len(data)/samplerate} sec")

        # Run AI Model (Mock Response)
        result = "The voice analysis suggests a neutral mood."  # Replace with AI logic
        return {"status": "success", "message": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get_result")
async def get_result():
    return {"message": "Mental health analysis completed!"}
