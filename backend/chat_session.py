# chat_session_routes.py

from fastapi import APIRouter, HTTPException, status, Header
from pydantic import BaseModel, EmailStr
from typing import List
from datetime import datetime
import uuid
from supabase import create_client, Client
from schemas import ChatMessage, ChatSessionRequest
import os
from dotenv import load_dotenv
import logging


# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not found in environment variables")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

router = APIRouter()

@router.post("/session/")
def save_chat_session(payload: ChatSessionRequest,  authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    try:
        token = authorization.split("Bearer ")[-1]
        
        print("Chat Session Payload: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",payload.json())

        # Set the Supabase session BEFORE any action
        supabase.auth.set_session(access_token=token, refresh_token=token)
        print(1)
        # Optional: Validate the token really belongs to this email
        user = supabase.auth.get_user()
        if not user or user.user.email != payload.user_email:
            raise HTTPException(status_code=403, detail="Access denied")

        print(2)
        data = {
            "id": str(uuid.uuid4()),
            "session_id": payload.session_id,
            "user_email": payload.user_email,
            "conversation": [msg.dict() for msg in payload.conversation],
            "started_at": payload.started_at.isoformat(),
            "ended_at": payload.ended_at.isoformat(),
            "conversation_date": payload.conversation_date
        }
        print(3, data)
        res = supabase.table("chat_sessions").insert(data).execute()
        print(4, res)
        if res.error:
            raise HTTPException(status_code=500, detail=res.error.message)

        return {"message": "Chat session saved successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
