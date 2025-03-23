import streamlit as st
import requests
from streamlit_chat import message
from dotenv import load_dotenv
import os
from datetime import datetime, timezone
import streamlit as st
import requests
import pandas as pd
import json
import io


# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")
SAVE_SESSION_URL = f"{API_URI}/session/"  # <-- new endpoint for saving full chat

def run(session):
    if not session:
        st.warning("Please log in to access your profile.")
        st.stop()
    
    st.subheader("📚 My Sessions")

    profile = st.session_state.get("profile")

    if not profile:
        st.warning("No profile found. Please login.")
    
    

    # Authenticated user details
    user_email = profile['email']
    access_token = session.get("access_token") 

    # Filters
    st.markdown("📅 Filter by Date")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date")
    end_date = col2.date_input("End Date")

    # Fetch sessions
    params = {
        "user_email": user_email,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(SAVE_SESSION_URL, params=params, headers=headers)
    sessions = response.json()

    # Filtered search inside conversations
    search_term = st.text_input("🔍 Search within sessions (text only)")

    filtered_sessions = []
    for session in sessions:
        if not search_term:
            filtered_sessions.append(session)
        else:
            for msg in session['conversation']:
                if search_term.lower() in msg['text'].lower():
                    filtered_sessions.append(session)
                    break

    # Export option
    if st.button("📤 Export Filtered Results"):
        if filtered_sessions:
            df = pd.DataFrame([
                {
                    "Session ID": s['session_id'],
                    "Started At": s['started_at'],
                    "Ended At": s['ended_at'],
                    "Total Messages": len(s['conversation'])
                } for s in filtered_sessions
            ])
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button("Download CSV", data=csv_buffer.getvalue(), file_name="filtered_sessions.csv", mime="text/csv")
        else:
            st.warning("No sessions to export.")

    # Display sessions
    st.markdown(f"🧾 Showing {len(filtered_sessions)} session(s)")
    for s in filtered_sessions:
        with st.expander(f"🗓️ {s['started_at']} - {s['ended_at']} (Messages: {len(s['conversation'])})"):
            for msg in s['conversation']:
                role = "🧠 AI" if not msg['is_user'] else "🧍 You"
                st.markdown(f"**{role}:** {msg['text']}")


