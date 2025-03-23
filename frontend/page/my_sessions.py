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

# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")
SAVE_SESSION_URL = f"{API_URI}/session/" # GET request
# Load a Unicode font (ensure .ttf file is available)
FONT_PATH = "DejaVuSans.ttf" 
BOLD_FONT_PATH = "DejaVuSans-Bold.ttf"

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
    
    # Toggle to generate PDF download button
    generate_pdf = st.checkbox("✅ Prepare PDF for Download")

    if generate_pdf:
        if filtered_sessions:
            pdf_data = generate_pdf_unicode(filtered_sessions)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_data,
                file_name=f"chat_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No sessions available to generate PDF.")



    # Display sessions
    st.markdown(f"🧾 Showing {len(filtered_sessions)} session(s)")
    for s in filtered_sessions:
        with st.expander(f"🗓️ {s['started_at']} - {s['ended_at']} (Messages: {len(s['conversation'])})"):
            for msg in s['conversation']:
                role = "🧠 AI" if not msg['is_user'] else "🧍 You"
                st.markdown(f"**{role}:** {msg['text']}")

  



# Define font paths
# FONT_PATH = "fonts/DejaVuSans.ttf"
# BOLD_FONT_PATH = "fonts/DejaVuSans-Bold.ttf"

def clean_text(text):
    return re.sub(r'[^\x00-\x7F\u00A0-\uFFFF]+', '', text)

def generate_pdf_unicode(session_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    usable_width = pdf.w - 2 * pdf.l_margin

    try:
        pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
        pdf.add_font("DejaVu", "B", BOLD_FONT_PATH, uni=True)
        pdf.set_font("DejaVu", size=12)
    except Exception as e:
        print("Font loading error:", e)
        pdf.set_font("Arial", size=12)

    for session in session_data:
        try:
            pdf.set_font("DejaVu", style='B', size=14)
            pdf.cell(usable_width, 10, txt=f"Session ID: {session['session_id']}", ln=True)
            pdf.set_font("DejaVu", size=12)
            pdf.cell(usable_width, 10, txt=f"Started At: {session['started_at']}", ln=True)
            pdf.cell(usable_width, 10, txt=f"Ended At: {session['ended_at']}", ln=True)
            pdf.cell(usable_width, 10, txt="Conversation:", ln=True)

            for message in session["conversation"]:
                sender = "You" if message["is_user"] else "Bot"
                raw_text = f"{sender}: {message['text']}"
                cleaned_text = clean_text(raw_text)

                # Handle long unbreakable words
                cleaned_text = re.sub(r'(\S{80,})', lambda m: '\n'.join(textwrap.wrap(m.group(0), 80)), cleaned_text)

                # Wrap the cleaned line for consistency
                wrapped_lines = textwrap.wrap(cleaned_text, width=100)

                # Always reset cursor to left margin
                pdf.set_x(pdf.l_margin)

                for line in wrapped_lines:
                    try:
                        pdf.multi_cell(w=usable_width, h=10, txt=line)
                        pdf.ln(1)  # Small vertical spacing between messages
                    except Exception as inner_e:
                        print("MultiCell error for line:", line)
                        print("Error:", inner_e)
                        pdf.set_x(pdf.l_margin)
                        pdf.cell(usable_width, 10, txt="[Unrenderable line]", ln=True)

            pdf.ln(5)

        except Exception as outer_e:
            print("Session-level error:", outer_e)
            pdf.set_font("Arial", size=12)
            pdf.set_x(pdf.l_margin)
            pdf.cell(usable_width, 10, txt="[Error rendering this session]", ln=True)

    pdf_output = BytesIO()
    try:
        pdf.output(pdf_output)
    except Exception as final_e:
        print("Final PDF output error:", final_e)
        return None

    pdf_output.seek(0)
    return pdf_output
