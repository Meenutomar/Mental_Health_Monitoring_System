from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SESSION_ANALYSIS_TABLE = "Session_Analysis"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not found in environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_chat_session(session_id: str):
    response = (
        supabase.table("Chat_Sessions")
        .select("*")
        .eq("id", session_id)
        .execute()
    )
    data = response.data
    return data[0] if data else None

def get_session_analysis(session_id: str):
    response = (
        supabase.table(SESSION_ANALYSIS_TABLE)
        .select("*")
        .eq("session_id", session_id)
        .execute()
    )
    data = response.data
    return data[0] if data else None

def insert_analysis(data: dict):
    response = supabase.table(SESSION_ANALYSIS_TABLE).insert(data).execute()
    return response.data[0]

def get_user_analyses(user_id: str):
    response = (
        supabase.table(SESSION_ANALYSIS_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data
