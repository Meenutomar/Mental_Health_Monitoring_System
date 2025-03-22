# schemas.py

from typing import List, Dict
from pydantic import BaseModel, EmailStr
from datetime import datetime

class ChatMessage(BaseModel):
    text: str
    is_user: bool

class ChatSessionRequest(BaseModel):
    session_id: str
    user_email: EmailStr
    conversation: List[ChatMessage]
    started_at: datetime
    ended_at: datetime
    conversation_date: str  # Format: YYYY-MM-DD
