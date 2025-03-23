from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import matplotlib.pyplot as plt
from datetime import datetime

import os
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

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from io import BytesIO

def generate_pdf(profile, analyses, logo_path=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterTitle', fontSize=16, leading=20, alignment=TA_LEFT, spaceAfter=20, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name='NormalText', fontSize=12, leading=15))

    story = []

    # Logo + Title side by side
    elements = []
    if logo_path and os.path.exists(logo_path):
        logo = Image(logo_path, width=50, height=50)
        elements.append(logo)
    title = Paragraph("Mental Health Progress Report", styles["CenterTitle"])
    elements.append(title)
    story.append(Table([elements], colWidths=[60, 400], hAlign='LEFT', style=[
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(Spacer(1, 20))

    # Patient Details + Profile Pic side by side
    details = [
        Paragraph(f"<b>Name:</b> {profile['name']}", styles["NormalText"]),
        Paragraph(f"<b>Age:</b> {profile['age']}", styles["NormalText"]),
        Paragraph(f"<b>Email:</b> {profile['email']}", styles["NormalText"])
    ]
    details_table = Table([[d] for d in details], style=[
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ])

    image_cell = ''
    profile_pic_path = profile['profile_pic_url']

    if profile_pic_path:
        profile_pic = Image(profile_pic_path, width=80, height=80)
        image_cell = profile_pic

    story.append(Table(
        [[details_table, image_cell]],
        colWidths=[350, 100],
        style=[
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (1, 0), (1, 0), 20)
        ]
    ))
    story.append(Spacer(1, 20))


   
    # Analyses
    for analysis in analyses:
        # If `created_at` is a string, convert it to datetime first:
        created_at = analysis['created_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        formatted_date = created_at.strftime("%Y-%m-%d")
        story.append(Paragraph(f"<b>Date:</b> {formatted_date}", styles["NormalText"]))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"<b>Score:</b> {analysis['score']}", styles["NormalText"]))
        story.append(Paragraph(f"<b>Emotions:</b> {analysis['summary']}", styles["NormalText"]))
        story.append(Paragraph(f"<b>Insights:</b> {analysis['change']}", styles["NormalText"]))
        story.append(Spacer(1, 15))

        # Score Graph as Bar Chart
        if analyses:
            try:
                scores = [a['score'] for a in analyses]
                dates = [datetime.fromisoformat(a['created_at']).strftime('%d %b') for a in analyses]

                plt.figure(figsize=(6, 3))
                plt.bar(dates, scores, color='skyblue', edgecolor='black')
                plt.title('Score Progress Over Time')
                plt.xlabel('Date')
                plt.ylabel('Score')
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', linewidth=0.5)

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
