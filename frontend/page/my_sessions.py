import streamlit as st
import requests
from streamlit_chat import message
from dotenv import load_dotenv
import os
from datetime import datetime, timezone
import streamlit as st
import requests
import pandas as pd
import io
from fpdf import FPDF
from io import BytesIO
import textwrap
import re
import my_sessions_pdf

# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")
SAVE_SESSION_URL = f"{API_URI}/session/" # GET request
# Load a Unicode font (ensure .ttf file is available)
#FONT_PATH = "DejaVuSans.ttf" 
#BOLD_FONT_PATH = "DejaVuSans-Bold.ttf"
# Define font paths
FONT_PATH = "fonts/DejaVuSans.ttf"
BOLD_FONT_PATH = "fonts/DejaVuSans-Bold.ttf"

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

    with st.expander("🔎 Filter Options", expanded=True):
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        search_term = st.text_input("Search within sessions")


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

    filtered_sessions = []
    for session in sessions:
        if not search_term:
            filtered_sessions.append(session)
        else:
            for msg in session['conversation']:
                if search_term.lower() in msg['text'].lower():
                    filtered_sessions.append(session)
                    break# Horizontal layout for export and download options

    col_pdf1, col_pdf2, col_pdf3 = st.columns([1.5, 2, 2])

    with col_pdf1:
        export_csv = st.button("📤 Export Results")

    with col_pdf2:
        generate_pdf = st.checkbox(" Prepare PDF for Download")

    with col_pdf3:
        if generate_pdf and filtered_sessions:
            pdf_data = my_sessions_pdf.generate_pdf_unicode (
                            session_data=filtered_sessions,
                            profile=profile,
                            logo_path="./assets/logo.png",
                            profile_pic_path=profile['profile_pic_url']  # or dynamic from user upload
                        )

            st.download_button(
                label="📄 Download PDF",
                data=pdf_data,
                file_name=f"chat_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

    # Export CSV logic
    if export_csv:
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

