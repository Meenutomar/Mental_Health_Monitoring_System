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
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")
ANALYZE_SESSION_URL = f"{API_URI}/analysis" 

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
                    st.markdown(f"**Emotions Detected:** {row['tone']}")
                    st.markdown(f"**Insights:** {row['change']}")
                    st.markdown(f"**Score:** {row['score']}")
                    st.markdown(f"**Analysis:** {row['summary']}")
            
            if st.button("📥 Download Progress Report"):
                pdf_buffer = generate_pdf(profile, analyses)
                st.download_button(
                    label="Download PDF",
                    data=pdf_buffer,
                    file_name="mental_health_report.pdf",
                    mime="application/pdf"
                )
        else:
            st.info("No pre-existing reports found.")
            if st.button("🔄 Generate Fresh"):
                analysis_payload = {
                    "session_id": st.session_state["chat_session_id"],
                    "user_id": profile["email"]
                }
                # Grab token from the session
                access_token = session.get("access_token") 

                # Send token in Authorization header
                headers = {"Authorization": f"Bearer {access_token}"}

                # After session saved to Supabase from frontend
                response = requests.post(ANALYZE_SESSION_URL, json=analysis_payload, headers=headers)
                if response.status_code == 200:
                    st.success("Analysis completed successfully.")
                    st.rerun()
                else:
                    st.error("Failed to analyze session.")
                
               
    
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
    

def generate_pdf(profile, analyses):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterTitle', fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=20, spaceBefore=10, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name='NormalText', fontSize=12, leading=15))

    story = []

    # Bold Title
    story.append(Paragraph("Mental Health Progress Report", styles["CenterTitle"]))
    story.append(Paragraph(f"<b><u>Patient Details:</u></b>", styles["NormalText"]))
    story.append(Paragraph(f"<b>Name:</b> {profile['name']}", styles["NormalText"]))
    story.append(Paragraph(f"<b>Age:</b> {profile['age']}", styles["NormalText"]))
    story.append(Paragraph(f"<b>Email:</b> {profile['email']}", styles["NormalText"]))
 

    for analysis in analyses:
        story.append(Paragraph(f"<b>Date:</b> {analysis['created_at']}", styles["NormalText"]))
        story.append(Paragraph(f"<b>Score:</b> {analysis['score']}", styles["NormalText"]))
        story.append(Paragraph(f"<b>Emotions:</b> {analysis['summary']}", styles["NormalText"]))
        story.append(Paragraph(f"<b>Insights:</b> {analysis['change']}", styles["NormalText"]))
        story.append(Spacer(1, 15))

    doc.build(story)
    buffer.seek(0)
    return buffer
