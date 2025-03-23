from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from reports.analyze_service import (
    get_chat_session, get_session_analysis, insert_analysis, get_user_analyses
)

from reports.llm_util import analyze_session_with_gemini

router = APIRouter()

class AnalysisRequest(BaseModel):
    session_id: str
    user_id: str

@router.post("/analysis")
def analyze_session(request: AnalysisRequest):
    print("Analyzing session...", request.session_id)
    existing = get_session_analysis(request.session_id)
    if existing:
        return existing

    session = get_chat_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    conversation = session.get("conversation")  # assumed to be a list of {"text", "is_user"}
    result = analyze_session_with_gemini(conversation)

    save_data = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "tone": result["tone"],
        "score": result["score"],
        "summary": result["summary"],
        "change": result["change"],
    }

    analysis = insert_analysis(save_data)
    return analysis


@router.get("/analysis/user/{user_id}")
def get_user_reports(user_id: str):
    return get_user_analyses(user_id)
