import streamlit as st
import PyPDF2
import pdfplumber
from io import BytesIO
from datetime import datetime
from nlp_processor import MeetingNLPProcessor
from export_utils import MeetingExporter
import tempfile
import os
import re
from pathlib import Path
from email_utils import send_summary_email, EmailConfigError
from audio_processing.transcribe import transcribe_audio
from audio_processing.diarize import diarize_audio
from audio_processing.transcript_parser import parse_transcript_with_timestamps, has_timestamp_format
from summarizer.summarize import chunk_transcript
from summarizer.bart_summarizer import summarize_chunks_bart, merge_summaries_text, summarize_global
from summarizer.structure_formatter import build_structure

st.set_page_config(
    page_title="AIMS - AI Meeting Summarizer",
    page_icon="📝",
    layout="wide"
)

def _as_bullets(text: str):
    try:
        raw = (text or "").replace("\n", " ").strip()
        if not raw:
            return []
        # simple sentence split; avoid heavy NLP for speed
        parts = [p.strip() for p in raw.replace("?", ".").replace("!", ".").split(".")]
        bullets = [p for p in parts if p]
        # cap to reasonable number for UI readability
        return bullets[:20]
    except Exception:
        return []

def _sanitize(s: str) -> str:
    try:
        import re
        # replace unicode block/box drawing characters with a standard small dash separator
        t = re.sub(r"[\u2580-\u259F\u2500-\u257F\u25A0-\u25FF]+", "-", s or "")
        # collapse long runs of dashes to '----'
        t = re.sub(r"-\s*-+", "----", t)
        t = re.sub(r"^-{5,}$", "----", t, flags=re.MULTILINE)
        return t.strip()
    except Exception:
        return (s or "").strip()
    
def _sanitize_for_export(structured):
    """
    Improved sanitization to ensure:
      - agenda becomes list of {'title': ...}
      - agenda titles are NOT duplicated into decisions or action items
      - decisions are short strings (no dicts)
      - action items are cleaned and not whole-summary text
    """
    import re
    if not structured or not isinstance(structured, dict):
        return structured

    # shallow copy (we will rebuild fields)
    data = structured.copy()

    # helper cleaners
    def clean_text(s):
        if not s:
            return ""
        s = re.sub(r"Speaker\s*\d*:?", "", str(s), flags=re.IGNORECASE)   # remove speaker labels
        s = re.sub(r"\b(ma|am|ok|umm+|uh+|almost done|done)\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"[\u2500-\u259F]+", "", s)  # remove box/line unicode
        s = re.sub(r"\s+", " ", s).strip()
        return s

    instruction_phrases = (
        "create a clear professional meeting summary",
        "format:",
        "transcript:",
        "speaker names or filler words",
        "meeting summary format",
    )

    def is_instruction(text: str) -> bool:
        t = (text or "").lower()
        return any(phrase in t for phrase in instruction_phrases)

    action_prefix_patterns = [
        r"^(?:the\s+)?concerned\s+staff\s+will\s+",
        r"^everyone\s+okay\s+with\s+that[\?\.]?\s*",
        r"^and\s+i\s+will\s+",
        r"^i\s+will\s+",
        r"^we\s+will\s+",
    ]

    def strip_action_prefix(text: str) -> str:
        if not text:
            return ""
        cleaned = text
        for pattern in action_prefix_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" -.,")
        return cleaned or text.strip()

    def compact_sentence(blob: str, limit: int = 220) -> str:
        if not blob:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", blob)
        chosen = ""
        for part in parts:
            candidate = part.strip(" -•")
            if not candidate or len(candidate.split()) < 3:
                continue
            chosen = candidate
            break
        if not chosen:
            chosen = blob.strip()
        if len(chosen) > limit:
            return chosen[:limit].rstrip() + "..."
        return chosen

    def clean_decision_text(text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"\b(agreed\.?|agreement)\b\.?", "", text, flags=re.IGNORECASE).strip(" -.," )
        return cleaned or text

    # ----- 1) Normalize agenda to list of {title:..}
    raw_ag = data.get("agenda", []) or []
    clean_ag = []
    for item in raw_ag:
        if isinstance(item, dict):
            title = item.get("title") or item.get("discussion") or ""
        else:
            title = str(item or "")
        title = clean_text(title)
        if title:
            clean_ag.append({"title": title})
    data["agenda"] = clean_ag

    # Build set of agenda titles for dedup checks (lowercased, short form)
    agenda_titles_set = set()
    for a in clean_ag:
        t = a.get("title","")
        if t:
            agenda_titles_set.add(t.lower())

    # get summary text for reference
    summary_text = clean_text(data.get("summary",""))

    # ----- 2) Clean decisions: keep only meaningful short strings
    raw_decisions = data.get("decisions", []) or []
    clean_decisions = []
    seen_dec = set()
    for d in raw_decisions:
        # accept dicts or strings
        if isinstance(d, dict):
            text = d.get("text") or d.get("decision") or ""
        else:
            text = str(d or "")
        text = clean_text(text)
        text = clean_decision_text(text)
        if not text or is_instruction(text):
            continue
        # drop if it's identical/contains an agenda title
        lowered = text.lower()
        if any(agt in lowered for agt in agenda_titles_set):
            continue
        # drop if it's too long (likely noise)
        if len(text) > 500:
            continue
        # dedupe
        if lowered in seen_dec:
            continue
        seen_dec.add(lowered)
        clean_decisions.append(text)
    data["decisions"] = clean_decisions

    # ----- 3) Clean action_items: ensure task/responsible/deadline structure and remove agenda leaks
    raw_actions = data.get("action_items", []) or []
    clean_actions = []
    seen_tasks = set()
    for a in raw_actions:
        # unify representation
        if isinstance(a, dict):
            task = a.get("task","") or a.get("text","") or ""
            responsible = a.get("responsible","") or a.get("owner","") or ""
            deadline = a.get("deadline","") or a.get("due","") or ""
        else:
            task = str(a or "")
            responsible = ""
            deadline = ""

        task = clean_text(task)
        task = strip_action_prefix(task)
        responsible = clean_text(responsible)
        deadline = clean_text(deadline)

        if not task or is_instruction(task):
            continue
        # drop if task is basically the whole summary or contains it (prevents whole-summary-in-task bug)
        if summary_text and len(task) > 120 and task in summary_text:
            continue
        # drop if task equals or contains an agenda title
        lowered_task = task.lower()
        if any(agt in lowered_task for agt in agenda_titles_set):
            # if it contains agenda title but has extra useful words, try to remove the agenda title chunk
            for agt in agenda_titles_set:
                if agt in lowered_task and len(lowered_task) > len(agt) + 10:
                    # attempt to strip the title substring
                    task = re.sub(re.escape(agt), "", task, flags=re.IGNORECASE).strip()
                    lowered_task = task.lower()
        # after attempted strip, if still matches agenda, skip
        if any(agt == lowered_task or agt in lowered_task for agt in agenda_titles_set):
            continue

        # drop extremely long noise
        if len(task) > 600:
            task = task[:400].rstrip() + "..."

        task = compact_sentence(task)

        # dedupe tasks
        if lowered_task in seen_tasks:
            continue
        seen_tasks.add(lowered_task)

        clean_actions.append({
            "task": task,
            "responsible": responsible,
            "deadline": deadline
        })

    data["action_items"] = clean_actions

    # ----- 4) Ensure attendees is list of dicts with name/role cleaned
    raw_att = data.get("attendees", []) or []
    clean_att = []
    for at in raw_att:
        if isinstance(at, dict):
            name = clean_text(at.get("name",""))
            role = clean_text(at.get("role",""))
        else:
            name = clean_text(at)
            role = ""
        if name:
            clean_att.append({"name": name, "role": role})
    data["attendees"] = clean_att

    # ----- 5) Finally, ensure other lists (keywords, entity_actions) are cleaned lightly
    if isinstance(data.get("keywords", None), list):
        data["keywords"] = [clean_text(k) for k in data.get("keywords", []) if clean_text(k)]
    if isinstance(data.get("entity_actions", None), list):
        cleaned_ea = []
        for ea in data.get("entity_actions", []):
            if not isinstance(ea, dict):
                continue
            ent = clean_text(ea.get("entity",""))
            label = clean_text(ea.get("label",""))
            action = clean_text(ea.get("action",""))
            obj = clean_text(ea.get("object",""))
            snippet = clean_text(ea.get("snippet",""))
            if ent:
                cleaned_ea.append({"entity":ent,"label":label,"action":action,"object":obj,"snippet":snippet})
        data["entity_actions"] = cleaned_ea

    return data


def _build_email_body(structured: dict) -> str:
    """
    Compose a plain-text email body with key sections for quick sharing.
    """
    if not structured:
        return ""

    metadata = structured.get("metadata", {})
    summary = _sanitize(structured.get("summary", ""))
    decisions = structured.get("decisions", []) or []
    action_items = structured.get("action_items", []) or []

    lines = []
    title = metadata.get("title") or "Meeting Summary"
    date = metadata.get("date") or datetime.now().strftime("%d/%m/%Y")
    venue = metadata.get("venue", "")

    lines.append(f"Title: {title}")
    lines.append(f"Date: {date}")
    if venue:
        lines.append(f"Venue: {venue}")

    if summary:
        lines.append("\nDiscussion Summary:")
        lines.append(summary)

    if decisions:
        lines.append("\nDecisions:")
        for dec in decisions:
            lines.append(f"- {dec}")

    if action_items:
        lines.append("\nAction Items:")
        for idx, item in enumerate(action_items, start=1):
            task = item.get("task", "").strip()
            if not task:
                continue
            responsible = item.get("responsible", "").strip()
            deadline = item.get("deadline", "").strip()
            line = f"{idx}. {task}"
            if responsible:
                line += f" — Owner: {responsible}"
            if deadline:
                line += f" (Due: {deadline})"
            lines.append(line)

    lines.append("\nGenerated via AIMS - AI Meeting Summarizer")
    return "\n".join(lines).strip()

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def main():
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'current_transcript' not in st.session_state:
        st.session_state.current_transcript = ""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Sidebar Navigation
    st.sidebar.title("📝 AIMS")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Upload & Transcribe", "Summary","Export"],
        index=["Home", "Upload & Transcribe", "Summary","Export"].index(st.session_state.current_page) if st.session_state.current_page in ["Home", "Upload & Transcribe", "Summary","Export"] else 0
    )
    
    st.session_state.current_page = page
    
    # Route to appropriate page
    if page == "Home":
        home_page()
    elif page == "Upload & Transcribe":
        upload_transcribe_page()
    elif page == "Summary":
        summary_page()
    elif page == "Export":
        export_page()

def home_page():
    st.title("📝 AIMS - AI Meeting Summarizer")
    st.markdown("### To Ease Your Meeting Minutes Creation Process")
    st.markdown("---")
    
    st.markdown("""
    ## Welcome to AIMS
    
    **AI Meeting Summarizer** helps you automatically generate structured meeting minutes from transcripts or audio recordings.
    
    ### Quick Start Guide
    
    1. **Upload & Transcribe**: Upload your meeting transcript (text/PDF) or audio file
    2. **Process**: Click "Process Transcript" to generate structured minutes
    3. **Review**: Check the Summary page to review and edit extracted information
    4. **Export**: Download as PDF/DOCX or email to attendees
    
    ### Features
    
    - 📄 **Multiple Input Formats**: Text, PDF, or Audio files
    - 🎤 **Audio Transcription**: Automatic transcription with speaker diarization
    - 🧠 **AI-Powered Summarization**: Extract key topics, decisions, and action items
    - 👥 **Attendee Detection**: Automatically identify meeting participants
    - 📧 **Email Integration**: Send summaries directly to attendees
    """)
    
    if st.session_state.processed_data:
        st.success("✅ You have processed meeting data. Navigate to 'Summary' to view it.")
    else:
        st.info("👈 Start by going to 'Upload & Transcribe' to upload your meeting transcript or audio file.")

def upload_transcribe_page():
    st.title("📤 Upload & Transcribe")
    st.markdown("---")
    
    st.sidebar.header("📄 Input Options")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Paste Text", "Upload PDF", "Upload Text File", "Upload Audio"]
    )

    transcript_text = ""
    audio_temp_path = None
    diarized = None

    if input_method == "Paste Text":
        st.sidebar.info("Paste your meeting transcript in the text area below")
        transcript_text = st.text_area(
            "Meeting Transcript",
            height=300,
            placeholder="Paste your meeting transcript here...")
    
    
    elif input_method == "Upload PDF":
        uploaded_file = st.sidebar.file_uploader("Upload PDF transcript", type=['pdf'])
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                transcript_text = extract_text_from_pdf(uploaded_file)
                st.success("✅ PDF text extracted successfully!")
    
    elif input_method == "Upload Text File":
        uploaded_file = st.sidebar.file_uploader("Upload text file", type=['txt'])
        if uploaded_file:
            transcript_text = uploaded_file.read().decode('utf-8')
            st.success("✅ Text file loaded successfully!")
    
    elif input_method == "Upload Audio":
        uploaded_audio = st.sidebar.file_uploader("Upload audio file (.mp3/.wav/.m4a)", type=['mp3','wav','m4a'])
        if uploaded_audio:
            st.sidebar.info("Uploading and saving audio for processing...")
            tmp_dir = os.path.join(tempfile.gettempdir(), "meeting_ai")
            os.makedirs(tmp_dir, exist_ok=True)

            st.info("Transcription may take several minutes depending on file length and model. Please wait...")
            with st.spinner("Transcribing audio (this can take a while)..."):
                # use tiny model for much faster test transcriptions
                audio_path, transcript, transcript_json = transcribe_audio(
                    uploaded_audio, tmp_dir=tmp_dir, model_name="tiny"
                )

            # clear spinner and show immediate status
            if not audio_path:
                st.error(f"Failed to save uploaded audio file. {transcript.get('error','')}")
            elif transcript.get("error"):
                st.error(f"Transcription error: {transcript.get('error')}")
                # expose saved transcript json for debugging if present
                if transcript_json and os.path.exists(transcript_json):
                    st.sidebar.markdown(f"Transcription saved: {transcript_json}")
            else:
                st.success("✅ Transcription complete. Running speaker diarization...")
                segments = transcript.get("segments", [])
                diarize_json = os.path.join(tmp_dir, Path(uploaded_audio.name).stem + "_diarized.json")
                try:
                    diarized = diarize_audio(audio_path, segments, out_json=diarize_json, use_pyannote=False)
                except Exception as e:
                    st.error(f"Diarization failed: {e}")
                    diarized = None

                # build raw transcript text for metadata extraction if diarization succeeded
                if diarized:
                    transcript_text = "\n".join([f"{s.get('speaker','Speaker')}: {s.get('text','')}" for s in diarized])
                    st.success("✅ Diarization complete.")
                    # save for debugging
                    if transcript_json and os.path.exists(transcript_json):
                        st.sidebar.markdown(f"Transcription saved: {transcript_json}")
                    if diarize_json and os.path.exists(diarize_json):
                        st.sidebar.markdown(f"Diarization saved: {diarize_json}")
                else:
                    # fallback to plain transcript text
                    transcript_text = transcript.get("text","")

    st.sidebar.markdown("---")
    
    # When processing, if diarized exists prefer it
    if st.sidebar.button("🔄 Process Transcript", type="primary", use_container_width=True):
        if (input_method == "Upload Audio" and diarized) or transcript_text.strip():
            with st.spinner("Processing transcript..."):
                # prefer diarized segments if available
                if diarized:
                    segments_for_summarizer = diarized
                    full_text = transcript_text
                elif has_timestamp_format(transcript_text):
                    # Parse transcript with timestamps and speaker names
                    segments_for_summarizer = parse_transcript_with_timestamps(transcript_text)
                    # Build full text with speaker labels for metadata extraction
                    full_text = "\n".join([f"{s.get('speaker','Speaker')}: {s.get('text','')}" for s in segments_for_summarizer])
                    if not segments_for_summarizer:
                        # Fallback if parsing failed
                        full_text = transcript_text
                        segments_for_summarizer = [{"speaker":"Speaker 1","start":0,"end":0,"text":full_text}]
                else:
                    # fall back to existing plain-text path: create simple segments
                    full_text = transcript_text
                    segments_for_summarizer = [{"speaker":"Speaker 1","start":0,"end":0,"text":full_text}]
                # create slightly larger chunks to reduce total summarization calls
                chunks = chunk_transcript(segments_for_summarizer, max_chars=2200)
                # use a lighter distilBART model for faster inference
                summaries = summarize_chunks_bart(
                    chunks,
                    model_name="sshleifer/distilbart-cnn-12-6",
                    device=-1
                )
                merged = merge_summaries_text(summaries)
                # produce a global, more coherent summary
                final_summary = summarize_global(
                    merged,
                    model_name="sshleifer/distilbart-cnn-12-6",
                    device=-1
                )
                final_summary = _sanitize(final_summary)
                # sanitize full text for metadata parsing/display
                structured = build_structure(segments_for_summarizer, final_summary, full_text)
                # 🛑 Ensure agenda does NOT merge into decisions/summary/action items
                if isinstance(structured.get("agenda"), list):
                    structured["agenda"] = [
                        {"title": a.get("title", "") if isinstance(a, dict) else str(a)}
                        for a in structured["agenda"]
                    ]

                # 🔥 SANITIZE HERE (IMPORTANT)
                st.session_state.processed_data = structured
                st.session_state.current_transcript = full_text
                st.success("✅ Transcript processed successfully! Navigate to 'Summary' to view results.")
                st.rerun()
        else:
            st.error("⚠️ Please provide a transcript or audio file first!")
    
    if st.sidebar.button("🗑️ Clear All", use_container_width=True):
        st.session_state.processed_data = None
        st.session_state.current_transcript = ""
        st.rerun()
    
    if transcript_text:
        st.markdown("### Preview")
        st.text_area("Transcript Preview", transcript_text[:500] + ("..." if len(transcript_text) > 500 else ""), height=200, disabled=True)

def summary_page():
    st.title("📋 Summary")
    st.markdown("---")
    
    if not st.session_state.processed_data:
        st.info("👈 No processed data found. Please go to 'Upload & Transcribe' to process a transcript first.")
        return
    
    data = st.session_state.processed_data
    
    st.markdown("## 📋 Generated Meeting Minutes")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    # -------------------- MEETING METADATA --------------------
    with col1:
        st.markdown("### 📌 Meeting Information")
        metadata = data.get('metadata', {})
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            title = st.text_input("Title", value=metadata.get('title') or "Meeting Summary", key="title_edit")
            date = st.text_input("Date", value=metadata.get('date') or datetime.now().strftime('%d/%m/%Y'), key="date_edit")
            time = st.text_input("Time", value=metadata.get('time') or "", key="time_edit")
        
        with info_col2:
            venue = st.text_input("Venue", value=metadata.get('venue') or "", key="venue_edit")
            organizer = st.text_input("Organizer", value=metadata.get('organizer') or "", key="organizer_edit")
            recorder = st.text_input("Recorder", value=metadata.get('recorder') or "", key="recorder_edit")
        
        data['metadata']['title'] = title
        data['metadata']['date'] = date
        data['metadata']['time'] = time
        data['metadata']['venue'] = venue
        data['metadata']['organizer'] = organizer
        data['metadata']['recorder'] = recorder
    
    # -------------------- STATS --------------------
    with col2:
        st.markdown("### 📊 Processing Stats")
        st.metric("Agenda Items", len(data.get('agenda', [])))
        st.metric("Decisions Identified", len(data.get('decisions', [])))
        st.metric("Action Items Found", len(data.get('action_items', [])))
        st.metric("Attendees Detected", len(data.get('attendees', [])))
    
    st.markdown("---")
    
    # -------------------- TABS --------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👥 Attendees", 
    "📌 Agenda", 
    "📝 Summary", 
    "✅ Decisions", 
    "📌 Action Items", 
    "📅 Next Meeting",
])

    
    # -------------------- TAB 1: ATTENDEES --------------------
    with tab1:
        st.markdown("### 👥 Attendees")
        attendees = data.get('attendees', [])
        if attendees:
            for i, attendee in enumerate(attendees):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    attendees[i]['name'] = st.text_input(
                        f"Name {i+1}", 
                        value=attendee.get('name', ''),
                        key=f"attendee_name_{i}"
                    )
                with col_b:
                    attendees[i]['role'] = st.text_input(
                        f"Role {i+1}", 
                        value=attendee.get('role', ''),
                        key=f"attendee_role_{i}"
                    )
            
            if st.button("➕ Add Attendee"):
                attendees.append({'name': '', 'role': ''})
                st.rerun()
        else:
            st.info("No attendees detected. Click below to add manually.")
            if st.button("➕ Add Attendee"):
                data['attendees'] = [{'name': '', 'role': ''}]
                st.rerun()
    
    # -------------------- TAB 2: AGENDA (Simplified Title-only UI) --------------------
    with tab2:
        st.markdown("### 📌 Agenda (Titles only)")

        # Ensure agenda is a list of dicts with 'title'
        agenda = data.get('agenda', [])
        if not isinstance(agenda, list):
            agenda = []

        if agenda:
            for i, item in enumerate(agenda):
                # Show only one input: Title
                st.markdown(f"#### Agenda Item {i+1}")
                new_title = st.text_input(
                    "Title",
                    value=item.get("title", "") if isinstance(item, dict) else str(item),
                    key=f"agenda_title_{i}"
                )

                # Update session storage: keep minimal structure
                data['agenda'][i] = {"title": new_title}

                # Delete button
                if st.button(f"🗑 Delete Agenda Item {i+1}", key=f"delete_agenda_{i}"):
                    data['agenda'].pop(i)
                    st.rerun()

                st.markdown("---")

            if st.button("➕ Add Agenda Item"):
                data['agenda'].append({"title": ""})
                st.rerun()
        else:
            st.info("No agenda items detected.")
            if st.button("➕ Add Agenda Item"):
                data['agenda'] = [{"title": ""}]
                st.rerun()

    # -------------------- TAB 3: SUMMARY (IMPROVED) --------------------
    with tab3:
        st.markdown("### 📝 Discussion Summary")

        raw_summary = _sanitize(data.get("summary", ""))

        st.markdown("#### 📘 Improved AI-Generated Summary Preview")

        # ----------------------
        # NEW STRUCTURED SUMMARY VIEW
        # ----------------------
        if raw_summary:
            # Split into paragraph + bullets if model already generated them
            lines = [l.strip() for l in raw_summary.split("\n") if l.strip()]
            
            paragraph_part = lines[0]
            bullet_points = [l for l in lines[1:] if l.startswith("-") or l.startswith("•")]

            # Paragraph
            st.markdown(
                f"<p style='text-align: justify; font-size: 16px;'>{paragraph_part}</p>",
                unsafe_allow_html=True
            )

            # Bullet Points
            if bullet_points:
                st.markdown("#### Key Discussion Points")
                for bp in bullet_points:
                    st.markdown(f"- {bp.lstrip('-• ').strip()}")

        else:
            st.info("No summary available yet.")

        st.markdown("---")

        # ----------------------
        # TEXT AREA FOR USER TO EDIT SOURCE SUMMARY
        # ----------------------
        st.markdown("#### ✏️ Edit AI Summary (affects final export)")
        updated_raw = st.text_area(
            "Edit Summary",
            value=raw_summary,
            height=250,
            key="summary_edit",
            help="Edit this summary if needed. This will be used in the exported PDF/DOCX."
        )

        data["summary"] = updated_raw

    # -------------------- TAB 4: DECISIONS --------------------
    with tab4:
        st.markdown("### ✅ Decisions")
        decisions = data.get('decisions', [])
        if decisions:
            for i, dec in enumerate(decisions):
                data['decisions'][i] = st.text_input(
                    f"Decision {i+1}",
                    value=dec,
                    key=f"decision_{i}"
                )
            if st.button("➕ Add Decision"):
                decisions.append("")
                st.rerun()
        else:
            st.info("No decisions detected.")
            if st.button("➕ Add Decision"):
                data['decisions'] = [""]
                st.rerun()

    # -------------------- TAB 5: ACTION ITEMS --------------------
    with tab5:
        st.markdown("### 📌 Action Items")
        action_items = data.get('action_items', [])
        if action_items:
            for i, action in enumerate(action_items):
                st.markdown(f"**Action Item {i+1}**")
                col1, col2, col3, col4 = st.columns([4, 2, 2, 1])
                
                with col1:
                    action_items[i]['task'] = st.text_area("Task", value=action.get('task',''), key=f"task_{i}", height=60)
                with col2:
                    action_items[i]['responsible'] = st.text_input("Responsible", value=action.get('responsible',''), key=f"responsible_{i}")
                with col3:
                    action_items[i]['deadline'] = st.text_input("Deadline", value=action.get('deadline',''), key=f"deadline_{i}")
                with col4:
                    action_items[i]['status'] = st.selectbox(
                        "Status",
                        ["Pending","In progress","Completed","Upcoming"],
                        index=["Pending","In progress","Completed","Upcoming"].index(action.get('status','Pending')),
                        key=f"status_{i}"
                    )
                st.markdown("---")
            
            if st.button("➕ Add Action Item"):
                action_items.append({'task':'','responsible':'','deadline':'','status':'Pending'})
                st.rerun()

        else:
            st.info("No action items found.")
            if st.button("➕ Add Action Item"):
                data['action_items'] = [{'task':'','responsible':'','deadline':'','status':'Pending'}]
                st.rerun()

    # -------------------- TAB 6: NEXT MEETING --------------------
    with tab6:
        st.markdown("### 📅 Next Meeting")
        next_meeting = data.get('next_meeting', {})
        
        col1, col2 = st.columns(2)
        with col1:
            next_meeting['date'] = st.text_input("Date", value=next_meeting.get('date') or "", key="next_date")
            next_meeting['time'] = st.text_input("Time", value=next_meeting.get('time') or "", key="next_time")
        with col2:
            next_meeting['venue'] = st.text_input("Venue", value=next_meeting.get('venue') or "", key="next_venue")
            next_meeting['agenda'] = st.text_area("Agenda", value=next_meeting.get('agenda') or "", key="next_agenda", height=100)

    st.markdown("---")
    st.markdown("### 📧 Send Minutes via Email")
    email_subject_default = data.get("metadata", {}).get("title") or "Meeting Summary"
    email_body_default = _build_email_body(data)

    with st.form("email_form"):
        recipients_raw = st.text_input(
            "Recipient emails",
            value="",
            placeholder="alice@example.com, bob@example.com",
        )
        subject_input = st.text_input(
            "Subject",
            value=email_subject_default,
        )
        body_input = st.text_area(
            "Email body",
            value=email_body_default,
            height=280,
        )
        submitted = st.form_submit_button("Send Email", type="primary")

        if submitted:
            recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
            if not recipients:
                st.error("Please enter at least one recipient email.")
            else:
                try:
                    with st.spinner("Sending email..."):
                        send_summary_email(subject_input or email_subject_default, body_input, recipients)
                    st.success("📨 Email sent successfully!")
                except EmailConfigError as e:
                    st.error(f"Email configuration error: {e}")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")

    st.caption("Configure SMTP_HOST/PORT/USER/PASS/SENDER in your environment before sending.")

def export_page():
    st.title("📥 Export")
    st.markdown("---")
    
    if not st.session_state.processed_data:
        st.info("👈 No processed data found. Please go to 'Upload & Transcribe' to process a transcript first.")
        return
    
    data = st.session_state.processed_data
    # 🔥 SANITIZE BEFORE EXPORT (defensive)
    data = _sanitize_for_export(data)
    st.session_state.processed_data = data
    st.markdown("## Export Options")
    
    # Initialise buffers in session state if not present
    if "pdf_buffer" not in st.session_state:
        st.session_state.pdf_buffer = None
    if "docx_buffer" not in st.session_state:
        st.session_state.docx_buffer = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 PDF Export")
        if st.button("📄 Generate PDF", type="primary", use_container_width=True):
            try:
                exporter = MeetingExporter()
                with st.spinner("Generating PDF..."):
                    st.session_state.pdf_buffer = exporter.export_to_pdf(data)
                st.success("✅ PDF generated. Use the download button below.")
            except Exception as e:
                st.error(f"PDF export failed: {e}")
                st.session_state.pdf_buffer = None
        
        if st.session_state.pdf_buffer:
            filename = f"Meeting_Minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label="⬇️ Download PDF",
                data=st.session_state.pdf_buffer,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True
            )
    
    with col2:
        st.markdown("### 📝 DOCX Export")
        if st.button("📝 Generate DOCX", type="primary", use_container_width=True):
            try:
                exporter = MeetingExporter()
                with st.spinner("Generating DOCX..."):
                    st.session_state.docx_buffer = exporter.export_to_docx(data)
                st.success("✅ DOCX generated. Use the download button below.")
            except Exception as e:
                st.error(f"DOCX export failed: {e}")
                st.session_state.docx_buffer = None
            
        if st.session_state.docx_buffer:
            filename = f"Meeting_Minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            st.download_button(
                label="⬇️ Download DOCX",
                data=st.session_state.docx_buffer,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
