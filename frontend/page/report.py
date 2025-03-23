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
from io import BytesIO
from reportlab.pdfgen import canvas

# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")
ANALYZE_SESSION_URL = f"{API_URI}/analysis" # GET request

def run(session):
    if not session:
        st.warning("Please log in to access your profile.")
        st.stop()
    
    st.subheader("📚 My Reports")

    profile = st.session_state.get("profile")

    if not profile:
        st.warning("No profile found. Please login.")
    else:
        user = profile['email']
        analyses = fetch_user_analyses(session, user)
        print('Analyses:', analyses)
        if analyses:
            df = pd.DataFrame(analyses)
            df['created_at'] = pd.to_datetime(df['created_at'])

            st.subheader("📈 Recovery Over Time")
            st.line_chart(df.set_index('created_at')['score'])

            st.subheader("🧠 Session Insights")
            for i, row in df.iterrows():
                with st.expander(f"Session on {row['created_at'].strftime('%Y-%m-%d')}"):
                    st.markdown(f"**Emotions Detected:** {row['emotions']}")
                    st.markdown(f"**Insights:** {row['insights']}")
                    st.markdown(f"**Score:** {row['score']}")
            
            if st.button("📥 Download Progress Report"):
                pdf_buffer = generate_pdf(analyses)
                st.download_button(
                    label="Download PDF",
                    data=pdf_buffer,
                    file_name="mental_health_report.pdf",
                    mime="application/pdf"
                )
        else:
            st.info("No reports found.")
    
def fetch_user_analyses(session, user):
    if not session:
        st.warning("Please log in to access your profile.")
    
    access_token = session.get("access_token") 
    headers = {
                "Authorization": f"Bearer {access_token}"
        }

    response = requests.get(f"{ANALYZE_SESSION_URL}/user/{user}", headers=headers)
 
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch analysis data.")
        return []
    


def generate_pdf(analyses):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica", 12)

    y = 800
    p.drawString(100, y, "Mental Health Progress Report")
    y -= 30

    for analysis in analyses:
        p.drawString(100, y, f"Date: {analysis['created_at']}")
        y -= 20
        p.drawString(100, y, f"Score: {analysis['score']}")
        y -= 20
        p.drawString(100, y, f"Emotions: {analysis['emotions']}")
        y -= 20
        p.drawString(100, y, f"Insights: {analysis['insights'][:60]}...")
        y -= 40
        if y < 100:
            p.showPage()
            y = 800

    p.save()
    buffer.seek(0)
    return buffer
