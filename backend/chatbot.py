from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

router = APIRouter()

# Configure AI model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Define request model
class MentalHealthRequest(BaseModel):
    name: str
    age: int
    message: str

@router.post("/chat")
async def mental_health_assessment(request: MentalHealthRequest):
    print(f"Received request from {request.name}, Age: {request.age}, Message: {request.message}")

    if request.message.lower() in ["hi", "hello", "start"]:
        return {"assessment": f"Hello {request.name}! 😊 How are you feeling today?"}

    # Construct AI prompt
    prompt = f"""
    You are a supportive AI therapist for mental health.
    - Engage in meaningful conversations.
    - Remember past responses and adapt accordingly.
    - Provide emotional support, but **do not diagnose**.
    - Ask open-ended questions to encourage discussion.
    - Never repeat steps; keep the conversation flowing.

    The user, {request.name}, {request.age} years old, says: "{request.message}".
    Respond in a supportive way:
    """

    try:
        response = model.generate_content(prompt)
        analysis = response.text  # Extract AI-generated text
        return {"assessment": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

