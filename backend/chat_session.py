# chat_session_routes.py

from fastapi import APIRouter, HTTPException, status, Header, Query
from datetime import datetime, timedelta
import uuid
from supabase import create_client, Client
from schemas import ChatMessage, ChatSessionRequest
import os
from dotenv import load_dotenv
import logging
from typing import Optional


# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not found in environment variables")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

router = APIRouter()

@router.post("/session/", status_code=201)
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

        data = {
            "session_id": payload.session_id,
            "user_email": payload.user_email,
            "conversation": [msg.model_dump() for msg in payload.conversation],
            "started_at": payload.started_at.isoformat(),
            "ended_at": payload.ended_at.isoformat(),
        }
        res = supabase.table("Chat_Session").insert(data).execute()
        print('Response from backend:', res)
        return {"message": "Chat session saved successfully!"}

    except Exception as e:
        print('Exception', e)
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/session/")
def get_user_sessions(
    user_email: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    authorization: str = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    try:
        token = authorization.split("Bearer ")[-1]
        supabase.auth.set_session(access_token=token, refresh_token=token)

        query = supabase.table("Chat_Session").select("*").eq("user_email", user_email)

        if start_date:
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.gte("started_at", start_datetime.isoformat())

        if end_date:
            # Add 1 day to include the full end date up to 23:59:59
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            query = query.lt("started_at", end_datetime.isoformat())


        response = query.order("started_at", desc=True).execute()

        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
