# export_utils.py (final fixed version)
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
from io import BytesIO
import re
import os


class MeetingExporter:
    def __init__(self, header_image_path="college_header.jpg"):
        self.header_image_path = header_image_path

    # --------------------------
    # Helpers / Cleaners
    # --------------------------
    @staticmethod
    def _sanitize(text: str) -> str:
        if not text:
            return ""
        t = re.sub(r"[\u2580-\u259F\u2500-\u257F\u25A0-\u25FF]+", "-", text)
        t = re.sub(r"\s+", " ", t)
        t = t.strip()
        return t

    @staticmethod
    def _clean_action_text(text: str) -> str:
        if not text:
            return ""
        t = re.sub(r"\b(ma|ma\.|am|am\.)\b", "", text, flags=re.IGNORECASE)
        t = re.sub(r"\s+", " ", t)
        t = t.strip(" .,-")
        return t

    # --------------------------
    # Rule-based sentence builders (NO MODEL)
    # --------------------------
    def _format_action_sentence(self, task: str, responsible: str = "", deadline: str = "") -> str:
        """
        Build a single natural sentence for an action item using simple rules:
        - If responsible exists: "<Responsible> will <task> [by <deadline>]."
        - If no responsible: "The concerned staff will <task> [by <deadline>]."
        - Capitalize first letter; ensure task is in infinitive-like form (best-effort).
        """
        task = self._clean_action_text(task).strip()
        responsible = self._sanitize(responsible).strip()
        deadline = self._sanitize(deadline).strip()

        if not task:
            return ""

        # make sure task is not sentence-starting with a capital duplicated; lower-case start for "will"
        # If task already starts with a verb like "prepare", we keep it; if it starts with "to prepare", strip leading "to "
        task_for_sentence = re.sub(r'^\s*to\s+', '', task, flags=re.IGNORECASE)

        # ensure the task ends without trailing punctuation
        task_for_sentence = task_for_sentence.rstrip(". ")

        if responsible:
            subject = responsible
        else:
            subject = "The concerned staff"

        sentence = f"{subject} will {task_for_sentence}"
        if deadline:
            sentence = f"{sentence} by {deadline}"
        sentence = sentence.strip()
        # Add final period if missing
        if not sentence.endswith("."):
            sentence += "."
        # Capitalize first char
        sentence = sentence[0].upper() + sentence[1:]
        return sentence

    def _formal_summary_from_text(self, summary_text: str) -> str:
        """
        Build a formal, single-paragraph summary from the raw summary text.
        This is deterministic and rule-based (no external model).
        """
        s = self._sanitize(summary_text)
        if not s:
            return ""

        # Split into sentences by punctuation, then clean and join as formal paragraph
        parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', s) if p.strip()]
        # Capitalize each part and ensure it ends with a period
        cleaned_parts = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # ensure punctuation at end
            if not re.search(r'[.!?]$', p):
                p = p + "."
            p = p[0].upper() + p[1:]
            cleaned_parts.append(p)

        # If very short, expand slightly with templated connectors
        if len(cleaned_parts) == 1:
            formal = f"The meeting was convened to discuss the following: {cleaned_parts[0]}"
        else:
            # join parts into a single paragraph with transitions for professionalism
            formal = "The meeting was convened to discuss the following points. "
            formal += " ".join(cleaned_parts)
        return formal

    # --------------------------
    # Header helpers
    # --------------------------
    def _add_docx_header(self, doc: Document):
        if self.header_image_path and os.path.exists(self.header_image_path):
            try:
                pic = doc.add_picture(self.header_image_path, width=Inches(6.5))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            except Exception:
                # if image fails, ignore silently
                pass

    def _add_pdf_header(self, story):
        if self.header_image_path and os.path.exists(self.header_image_path):
            try:
                img = RLImage(self.header_image_path, width=6.5 * inch)
                story.append(img)
                story.append(Spacer(1, 12))
            except Exception:
                pass

    # --------------------------
    # DOCX Export
    # --------------------------
    def export_to_docx(self, meeting_data):
        doc = Document()

        # Header image
        self._add_docx_header(doc)

        # Title
        title_heading = doc.add_heading("Minutes of Meeting", 0)
        title_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()

        metadata = meeting_data.get("metadata", {})
        if metadata.get("title"):
            p = doc.add_paragraph()
            p.add_run("Title: ").bold = True
            p.add_run(str(metadata.get("title")))
        if metadata.get("date"):
            p = doc.add_paragraph()
            p.add_run("Date: ").bold = True
            p.add_run(str(metadata.get("date")))
        if metadata.get("time"):
            p = doc.add_paragraph()
            p.add_run("Time: ").bold = True
            p.add_run(str(metadata.get("time")))
        if metadata.get("venue"):
            p = doc.add_paragraph()
            p.add_run("Venue: ").bold = True
            p.add_run(str(metadata.get("venue")))
        if metadata.get("organizer"):
            p = doc.add_paragraph()
            p.add_run("Organizer: ").bold = True
            p.add_run(str(metadata.get("organizer")))
        if metadata.get("recorder"):
            p = doc.add_paragraph()
            p.add_run("Recorder: ").bold = True
            p.add_run(str(metadata.get("recorder")))

        doc.add_paragraph("----")

        # Attendees
        attendees = meeting_data.get("attendees", [])
        if attendees:
            doc.add_heading("Attendees", level=2)
            for a in attendees:
                name = self._sanitize(a.get("name", ""))
                role = self._sanitize(a.get("role", ""))
                if role:
                    doc.add_paragraph(f"{name} – {role}", style="List Bullet")
                else:
                    doc.add_paragraph(name, style="List Bullet")
            doc.add_paragraph("----")

        # Agenda table
        agenda = meeting_data.get("agenda", [])
        if agenda:
            doc.add_heading("Agenda", level=2)
            table = doc.add_table(rows=1, cols=3)
            table.style = "Light Grid Accent 1"
            hdr = table.rows[0].cells
            hdr[0].text = "Agenda No."
            hdr[1].text = "Discussion & Action to be taken"
            hdr[2].text = "Responsibility"
            # bold header
            for cell in hdr:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.bold = True
            # rows
            for item in agenda:
                row = table.add_row().cells
                row[0].text = str(item.get("no", ""))
                row[1].text = self._sanitize(item.get("discussion", ""))
                row[2].text = self._sanitize(item.get("responsibility", ""))
            doc.add_paragraph("----")

        # Discussion Summary (formal)
        summary = meeting_data.get("summary", "")
        if summary:
            doc.add_heading("Discussion Summary", level=2)
            formal = self._formal_summary_from_text(summary)
            doc.add_paragraph(formal)
            doc.add_paragraph("----")

        # Decisions
        decisions = meeting_data.get("decisions", [])
        if decisions:
            doc.add_heading("Decisions", level=2)
            for d in decisions:
                doc.add_paragraph(self._sanitize(d), style="List Bullet")
            doc.add_paragraph("----")

        # Action Items (rule-based sentences)
        action_items = meeting_data.get("action_items", [])
        if action_items:
            doc.add_heading("Action Items", level=2)
            for a in action_items:
                task = a.get("task", "")
                responsible = a.get("responsible", "")
                deadline = a.get("deadline", "")
                sentence = self._format_action_sentence(task, responsible, deadline)
                if sentence:
                    doc.add_paragraph(f"• {sentence}")
            doc.add_paragraph("----")

        # Next meeting
        next_meeting = meeting_data.get("next_meeting", {})
        if any(next_meeting.values()):
            doc.add_heading("Next Meeting", level=2)
            if next_meeting.get("date"):
                p = doc.add_paragraph(); p.add_run("Date: ").bold = True; p.add_run(str(next_meeting.get("date")))
            if next_meeting.get("time"):
                p = doc.add_paragraph(); p.add_run("Time: ").bold = True; p.add_run(str(next_meeting.get("time")))
            if next_meeting.get("venue"):
                p = doc.add_paragraph(); p.add_run("Venue: ").bold = True; p.add_run(str(next_meeting.get("venue")))
            if next_meeting.get("agenda"):
                p = doc.add_paragraph(); p.add_run("Agenda: ").bold = True; p.add_run(str(next_meeting.get("agenda")))
            doc.add_paragraph("─" * 50)

        # Closing note
        doc.add_heading("Closing Note", level=2)
        closing = f"Meeting minutes generated on {datetime.now().strftime('%d/%m/%Y at %H:%M')}."
        doc.add_paragraph(closing)

        buf = BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf

    # --------------------------
    # PDF Export
    # --------------------------
    def export_to_pdf(self, meeting_data):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=20)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="TitleStyle", fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=18))
        styles.add(ParagraphStyle(name="HeadingStyle", parent=styles["Heading2"], fontSize=14, leading=16, spaceAfter=12))
        styles.add(ParagraphStyle(name="BodyStyle", parent=styles["BodyText"], fontSize=11, leading=14, spaceAfter=6))
        styles.add(ParagraphStyle(name="TableBody", parent=styles["BodyText"], fontSize=10, leading=12))

        story = []
        # header image
        self._add_pdf_header(story)

        # title
        story.append(Paragraph("Minutes of Meeting", styles["TitleStyle"]))
        story.append(Spacer(1, 12))

        # metadata
        metadata = meeting_data.get("metadata", {})
        def add_meta(label, key):
            if metadata.get(key):
                story.append(Paragraph(f"<b>{label}:</b> {self._sanitize(metadata.get(key))}", styles["BodyStyle"]))
        add_meta("Title", "title")
        add_meta("Date", "date")
        add_meta("Time", "time")
        add_meta("Venue", "venue")
        add_meta("Organizer", "organizer")
        add_meta("Recorder", "recorder")
        story.append(Spacer(1, 12))
        story.append(Paragraph("----", styles["BodyStyle"]))
        story.append(Spacer(1, 12))

        # attendees
        attendees = meeting_data.get("attendees", [])
        if attendees:
            story.append(Paragraph("Attendees", styles["HeadingStyle"]))
            for a in attendees:
                name = self._sanitize(a.get("name", ""))
                role = self._sanitize(a.get("role", ""))
                if role:
                    story.append(Paragraph(f"• {name} – {role}", styles["BodyStyle"]))
                else:
                    story.append(Paragraph(f"• {name}", styles["BodyStyle"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph("----", styles["BodyStyle"]))
            story.append(Spacer(1, 12))

        # agenda table
        agenda = meeting_data.get("agenda", [])
        if agenda:
            story.append(Paragraph("Agenda", styles["HeadingStyle"]))
            table_data = [["Agenda No.", "Discussion & Action to be taken", "Responsibility"]]
            for item in agenda:
                table_data.append([str(item.get("no","")), self._sanitize(item.get("discussion","")), self._sanitize(item.get("responsibility",""))])
            table = Table(table_data, colWidths=[60, 330, 110])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E8EAF6")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                ('ALIGN', (0,0), (-1,0), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 10),
                ('GRID', (0,0), (-1,-1), 0.7, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,1), (-1,-1), 10),
                ('LEFTPADDING', (0,0), (-1,-1), 4),
                ('RIGHTPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(table)
            story.append(Spacer(1, 18))
            story.append(Paragraph("----", styles["BodyStyle"]))
            story.append(Spacer(1, 12))

        # discussion summary (formal)
        summary = meeting_data.get("summary", "")
        if summary:
            story.append(Paragraph("Discussion Summary", styles["HeadingStyle"]))
            formal = self._formal_summary_from_text(summary)
            story.append(Paragraph(formal, styles["BodyStyle"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph("----", styles["BodyStyle"]))
            story.append(Spacer(1, 12))

        # decisions
        decisions = meeting_data.get("decisions", [])
        if decisions:
            story.append(Paragraph("Decisions", styles["HeadingStyle"]))
            for d in decisions:
                story.append(Paragraph(f"• {self._sanitize(d)}", styles["BodyStyle"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph("----", styles["BodyStyle"]))
            story.append(Spacer(1, 12))

        # action items (rule-based)
        action_items = meeting_data.get("action_items", [])
        if action_items:
            story.append(Paragraph("Action Items", styles["HeadingStyle"]))
            for a in action_items:
                task = a.get("task","")
                responsible = a.get("responsible","")
                deadline = a.get("deadline","")
                sentence = self._format_action_sentence(task, responsible, deadline)
                if sentence:
                    story.append(Paragraph(f"• {sentence}", styles["BodyStyle"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph("----", styles["BodyStyle"]))
            story.append(Spacer(1, 12))

        # next meeting
        next_meeting = meeting_data.get("next_meeting", {})
        if any(next_meeting.values()):
            story.append(Paragraph("Next Meeting", styles["HeadingStyle"]))
            if next_meeting.get("date"):
                story.append(Paragraph(f"<b>Date:</b> {self._sanitize(next_meeting.get('date'))}", styles["BodyStyle"]))
            if next_meeting.get("time"):
                story.append(Paragraph(f"<b>Time:</b> {self._sanitize(next_meeting.get('time'))}", styles["BodyStyle"]))
            if next_meeting.get("venue"):
                story.append(Paragraph(f"<b>Venue:</b> {self._sanitize(next_meeting.get('venue'))}", styles["BodyStyle"]))
            if next_meeting.get("agenda"):
                story.append(Paragraph(f"<b>Agenda:</b> {self._sanitize(next_meeting.get('agenda'))}", styles["BodyStyle"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph("─" * 80, styles["BodyStyle"]))
            story.append(Spacer(1, 12))

        # closing
        story.append(Paragraph("Closing Note", styles["HeadingStyle"]))
        closing = f"Meeting minutes generated on {datetime.now().strftime('%d/%m/%Y at %H:%M')}."
        story.append(Paragraph(closing, styles["BodyStyle"]))

        doc.build(story)
        buffer.seek(0)
        return buffer
