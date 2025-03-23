import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure AI model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def analyze_session_with_gemini(conversation: list):
    prompt = (
        "You're a mental health assistant. Analyze the following conversation:\n\n"
    )
    for msg in conversation:
        speaker = "User" if msg.get("is_user") else "AI"
        prompt += f"{speaker}: {msg['text']}\n"

    prompt += (
        "\nReturn a JSON like this:\n"
        '{\n'
        '  "tone": "positive",\n'
        '  "score": 72,\n'
        '  "summary": "The user showed improvement...",\n'
        '  "change": "User appears calmer than previous session."\n'
        '}'
    )

    response = model.generate_content(prompt)
    return json.loads(response.text)
