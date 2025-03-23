from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import matplotlib.pyplot as plt
from datetime import datetime

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


def generate_pdf(profile, analyses, logo_path=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterTitle', fontSize=16, leading=20, alignment=TA_CENTER,
                              spaceAfter=20, spaceBefore=10, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name='NormalText', fontSize=12, leading=15))

    story = []

    # Profile Picture (Optional)
    if logo_path:
        try:
            img = Image(logo_path, width=100, height=100)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 15))
        except Exception as e:
            story.append(Paragraph(f"<i>Could not load logo picture: {e}</i>", styles["NormalText"]))
    # Title
    story.append(Paragraph("Mental Health Progress Report", styles["CenterTitle"]))
    profile_pic_path = profile['profile_pic_url']
    # Profile Picture (Optional)
    if profile_pic_path:
        try:
            img = Image(profile_pic_path, width=100, height=100)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 15))
        except Exception as e:
            story.append(Paragraph(f"<i>Could not load profile picture: {e}</i>", styles["NormalText"]))

    # Patient Details
    story.append(Paragraph(f"<b><u>Patient Details:</u></b>", styles["NormalText"]))
    story.append(Paragraph(f"<b>Name:</b> {profile['name']}", styles["NormalText"]))
    story.append(Paragraph(f"<b>Age:</b> {profile['age']}", styles["NormalText"]))
    story.append(Paragraph(f"<b>Email:</b> {profile['email']}", styles["NormalText"]))
    story.append(Spacer(1, 10))

    # Analysis Data
    for analysis in analyses:
        story.append(Paragraph(f"<b>Date:</b> {analysis['created_at']}", styles["NormalText"]))
        story.append(Paragraph(f"<b>Score:</b> {analysis['score']}", styles["NormalText"]))
        story.append(Paragraph(f"<b>Emotions:</b> {analysis['summary']}", styles["NormalText"]))
        story.append(Paragraph(f"<b>Insights:</b> {analysis['change']}", styles["NormalText"]))
        story.append(Spacer(1, 15))

    # Score Graph
    if analyses:
        try:
            scores = [a['score'] for a in analyses]
            dates = [datetime.fromisoformat(a['created_at']).strftime('%d %b') for a in analyses]

            plt.figure(figsize=(6, 3))
            plt.plot(dates, scores, marker='o', color='blue', linewidth=2)
            plt.title('Score Progress Over Time')
            plt.xlabel('Date')
            plt.ylabel('Score')
            plt.grid(True)

            graph_buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(graph_buffer, format='PNG')
            plt.close()
            graph_buffer.seek(0)

            story.append(Paragraph("<b>Progress Graph:</b>", styles["NormalText"]))
            story.append(Image(graph_buffer, width=400, height=200))
            story.append(Spacer(1, 20))
        except Exception as e:
            story.append(Paragraph(f"<i>Could not render graph: {e}</i>", styles["NormalText"]))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer
