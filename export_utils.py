from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from datetime import datetime

class MeetingExporter:
    def __init__(self):
        pass
    
    @staticmethod
    def _sanitize(s: str) -> str:
        try:
            import re
            t = re.sub(r"[\u2580-\u259F\u2500-\u257F\u25A0-\u25FF]+", "-", s or "")
            t = re.sub(r"-\s*-+", "----", t)
            t = re.sub(r"^-{5,}$", "----", t, flags=re.MULTILINE)
            return t
        except Exception:
            return s or ""
    
    def export_to_docx(self, meeting_data):
        doc = Document()
        
        title_heading = doc.add_heading('AIMS – Meeting Summary', 0)
        title_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_paragraph()
        
        metadata = meeting_data.get('metadata', {})
        if metadata.get('title'):
            p = doc.add_paragraph()
            p.add_run('Title: ').bold = True
            p.add_run(metadata['title'])
        
        if metadata.get('date'):
            p = doc.add_paragraph()
            p.add_run('Date: ').bold = True
            p.add_run(metadata['date'])
        
        if metadata.get('time'):
            p = doc.add_paragraph()
            p.add_run('Time: ').bold = True
            p.add_run(metadata['time'])
        
        if metadata.get('venue'):
            p = doc.add_paragraph()
            p.add_run('Venue: ').bold = True
            p.add_run(metadata['venue'])
        
        if metadata.get('organizer'):
            p = doc.add_paragraph()
            p.add_run('Organizer: ').bold = True
            p.add_run(metadata['organizer'])
        
        if metadata.get('recorder'):
            p = doc.add_paragraph()
            p.add_run('Recorder: ').bold = True
            p.add_run(metadata['recorder'])
        
        doc.add_paragraph('----')
        
        attendees = meeting_data.get('attendees', [])
        if attendees:
            doc.add_heading('Attendees', level=2)
            for attendee in attendees:
                name = attendee.get('name', '')
                role = attendee.get('role', '')
                if role:
                    doc.add_paragraph(f"{name} – {role}", style='List Bullet')
                else:
                    doc.add_paragraph(name, style='List Bullet')
            doc.add_paragraph('----')
        
        key_topics = meeting_data.get('key_topics', [])
        if key_topics:
            doc.add_heading('Key Topics', level=2)
            for topic in key_topics:
                doc.add_paragraph(topic.title(), style='List Bullet')
            doc.add_paragraph('----')
        
        summary = meeting_data.get('summary', '')
        if summary:
            doc.add_heading('Discussion Summary', level=2)
            doc.add_paragraph(summary)
            doc.add_paragraph('----')
        
        decisions = meeting_data.get('decisions', [])
        if decisions:
            doc.add_heading('Decisions', level=2)
            for decision in decisions:
                doc.add_paragraph(decision, style='List Bullet')
            doc.add_paragraph('----')
        
        action_items = meeting_data.get('action_items', [])
        if action_items:
            doc.add_heading('Action Items', level=2)
            
            table = doc.add_table(rows=1, cols=4)
            table.style = 'Light Grid Accent 1'
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Task'
            hdr_cells[1].text = 'Responsible Person'
            hdr_cells[2].text = 'Deadline'
            hdr_cells[3].text = 'Status'
            
            for cell in hdr_cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
            
            for action in action_items:
                row_cells = table.add_row().cells
                row_cells[0].text = action.get('task', '')
                row_cells[1].text = action.get('responsible', 'Not specified')
                row_cells[2].text = action.get('deadline', 'Not specified')
                row_cells[3].text = action.get('status', 'Pending')
            
            doc.add_paragraph('----')
        
        next_meeting = meeting_data.get('next_meeting', {})
        if any(next_meeting.values()):
            doc.add_heading('Next Meeting', level=2)
            if next_meeting.get('date'):
                p = doc.add_paragraph()
                p.add_run('Date: ').bold = True
                p.add_run(next_meeting['date'])
            if next_meeting.get('time'):
                p = doc.add_paragraph()
                p.add_run('Time: ').bold = True
                p.add_run(next_meeting['time'])
            if next_meeting.get('venue'):
                p = doc.add_paragraph()
                p.add_run('Venue: ').bold = True
                p.add_run(next_meeting['venue'])
            if next_meeting.get('agenda'):
                p = doc.add_paragraph()
                p.add_run('Agenda: ').bold = True
                p.add_run(next_meeting['agenda'])
            doc.add_paragraph('─' * 50)
        
        doc.add_heading('Closing Note', level=2)
        closing_text = f"Meeting minutes generated by AIMS - AI Meeting Summarizer on {datetime.now().strftime('%d/%m/%Y at %H:%M')}."
        doc.add_paragraph(self._sanitize(closing_text))
        
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer
    
    def export_to_pdf(self, meeting_data):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CustomTitle',
                                  parent=styles['Heading1'],
                                  fontSize=18,
                                  textColor=colors.HexColor('#1f4788'),
                                  spaceAfter=30,
                                  alignment=TA_CENTER,
                                  fontName='Helvetica-Bold'))
        
        styles.add(ParagraphStyle(name='SectionHeading',
                                  parent=styles['Heading2'],
                                  fontSize=14,
                                  textColor=colors.HexColor('#2a5298'),
                                  spaceAfter=12,
                                  spaceBefore=12,
                                  fontName='Helvetica-Bold'))
        
        styles.add(ParagraphStyle(name='CustomBody',
                                  parent=styles['BodyText'],
                                  fontSize=11,
                                  spaceAfter=6,
                                  fontName='Helvetica'))
        
        story = []
        
        title = Paragraph("AIMS – Meeting Summary", styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        metadata = meeting_data.get('metadata', {})
        if metadata.get('title'):
            story.append(Paragraph(f"<b>Title:</b> {metadata['title']}", styles['CustomBody']))
        if metadata.get('date'):
            story.append(Paragraph(f"<b>Date:</b> {metadata['date']}", styles['CustomBody']))
        if metadata.get('time'):
            story.append(Paragraph(f"<b>Time:</b> {metadata['time']}", styles['CustomBody']))
        if metadata.get('venue'):
            story.append(Paragraph(f"<b>Venue:</b> {metadata['venue']}", styles['CustomBody']))
        if metadata.get('organizer'):
            story.append(Paragraph(f"<b>Organizer:</b> {metadata['organizer']}", styles['CustomBody']))
        if metadata.get('recorder'):
            story.append(Paragraph(f"<b>Recorder:</b> {metadata['recorder']}", styles['CustomBody']))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("----", styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        attendees = meeting_data.get('attendees', [])
        if attendees:
            story.append(Paragraph("Attendees", styles['SectionHeading']))
            for attendee in attendees:
                name = attendee.get('name', '')
                role = attendee.get('role', '')
                if role:
                    story.append(Paragraph(f"• {name} – {role}", styles['CustomBody']))
                else:
                    story.append(Paragraph(f"• {name}", styles['CustomBody']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("----", styles['CustomBody']))
            story.append(Spacer(1, 12))
        
        key_topics = meeting_data.get('key_topics', [])
        if key_topics:
            story.append(Paragraph("Key Topics", styles['SectionHeading']))
            for topic in key_topics:
                story.append(Paragraph(f"• {topic.title()}", styles['CustomBody']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("----", styles['CustomBody']))
            story.append(Spacer(1, 12))
        
        summary = meeting_data.get('summary', '')
        if summary:
            story.append(Paragraph("Discussion Summary", styles['SectionHeading']))
            story.append(Paragraph(summary, styles['CustomBody']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("----", styles['CustomBody']))
            story.append(Spacer(1, 12))
        
        decisions = meeting_data.get('decisions', [])
        if decisions:
            story.append(Paragraph("Decisions", styles['SectionHeading']))
            for decision in decisions:
                story.append(Paragraph(f"• {decision}", styles['CustomBody']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("----", styles['CustomBody']))
            story.append(Spacer(1, 12))
        
        action_items = meeting_data.get('action_items', [])
        if action_items:
            story.append(Paragraph("Action Items", styles['SectionHeading']))
            
            table_data = [['Task', 'Responsible', 'Deadline', 'Status']]
            for action in action_items:
                table_data.append([
                    action.get('task', '')[:50],
                    action.get('responsible', 'Not specified'),
                    action.get('deadline', 'Not specified'),
                    action.get('status', 'Pending')
                ])
            
            table = Table(table_data, colWidths=[3*inch, 1.2*inch, 1.2*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a5298')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 12))
            story.append(Paragraph("----", styles['CustomBody']))
            story.append(Spacer(1, 12))
        
        next_meeting = meeting_data.get('next_meeting', {})
        if any(next_meeting.values()):
            story.append(Paragraph("Next Meeting", styles['SectionHeading']))
            if next_meeting.get('date'):
                story.append(Paragraph(f"<b>Date:</b> {next_meeting['date']}", styles['CustomBody']))
            if next_meeting.get('time'):
                story.append(Paragraph(f"<b>Time:</b> {next_meeting['time']}", styles['CustomBody']))
            if next_meeting.get('venue'):
                story.append(Paragraph(f"<b>Venue:</b> {next_meeting['venue']}", styles['CustomBody']))
            if next_meeting.get('agenda'):
                story.append(Paragraph(f"<b>Agenda:</b> {next_meeting['agenda']}", styles['CustomBody']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("─" * 80, styles['CustomBody']))
            story.append(Spacer(1, 12))
        
        story.append(Paragraph("Closing Note", styles['SectionHeading']))
        closing_text = f"Meeting minutes generated by AIMS - AI Meeting Summarizer on {datetime.now().strftime('%d/%m/%Y at %H:%M')}."
        story.append(Paragraph(closing_text, styles['CustomBody']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
