import pandas as pd
import io
from fpdf import FPDF
from io import BytesIO
import textwrap
import re
import os

FONT_PATH = "fonts/DejaVuSans.ttf"
BOLD_FONT_PATH = "fonts/DejaVuSans-Bold.ttf"

def clean_text(text):
    return re.sub(r'[^\x00-\x7F\u00A0-\uFFFF]+', '', text)

def generate_pdf_unicode(session_data, profile, logo_path, profile_pic_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    usable_width = pdf.w - 2 * pdf.l_margin

    # Load fonts
    try:
        pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
        pdf.add_font("DejaVu", "B", BOLD_FONT_PATH, uni=True)
        pdf.set_font("DejaVu", size=12)
    except Exception as e:
        print("Font loading error:", e)
        pdf.set_font("Arial", size=12)

    # --- Header Section: Logo + Profile Picture + Details ---
    try:
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=pdf.l_margin, y=10, w=30)

        if profile_pic_path:
            pdf.image(profile_pic_path, x=pdf.w - pdf.r_margin - 30, y=10, w=30)

        pdf.set_xy(pdf.l_margin + 35, 10)
        pdf.set_font("DejaVu", style='B', size=14)
        pdf.cell(0, 10, txt="Mental Health Session Report", ln=True)

        pdf.set_font("DejaVu", size=12)
        pdf.set_xy(pdf.l_margin + 35, 20)
        pdf.cell(0, 10, txt=f"Name: {profile.get('name', '')}", ln=True)

        pdf.set_xy(pdf.l_margin + 35, 30)
        pdf.cell(0, 10, txt=f"Age: {profile.get('age', '')}    Email: {profile.get('email', '')}", ln=True)

        pdf.ln(30)
    except Exception as header_e:
        print("Header render error:", header_e)

    # --- Session Data ---
    for session in session_data:
        try:
            pdf.set_font("DejaVu", style='B', size=12)
            pdf.cell(usable_width, 10, txt=f"Session ID: {session['session_id']}", ln=True)
            pdf.set_font("DejaVu", size=10)
            pdf.cell(usable_width, 10, txt=f"Started At: {session['started_at']}", ln=True)
            pdf.cell(usable_width, 10, txt=f"Ended At: {session['ended_at']}", ln=True)
            pdf.cell(usable_width, 10, txt="Conversation:", ln=True)

            for message in session["conversation"]:
                sender = "You" if message["is_user"] else "Bot"
                raw_text = f"{sender}: {message['text']}"
                cleaned_text = clean_text(raw_text)
                cleaned_text = re.sub(r'(\S{80,})', lambda m: '\n'.join(textwrap.wrap(m.group(0), 80)), cleaned_text)
                wrapped_lines = textwrap.wrap(cleaned_text, width=100)

                pdf.set_x(pdf.l_margin)
                for line in wrapped_lines:
                    try:
                        pdf.multi_cell(w=usable_width, h=10, txt=line)
                        pdf.ln(1)
                    except Exception as inner_e:
                        print("MultiCell error for line:", line)
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
