import streamlit as st
import PyPDF2
import pdfplumber
from io import BytesIO
from datetime import datetime
from nlp_processor import MeetingNLPProcessor
from export_utils import MeetingExporter
import tempfile
import os
from pathlib import Path
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
    st.title("📝 AIMS - AI Meeting Summarizer")
    st.markdown("To Ease Your Meeting Minutes Creation Process")
    st.markdown("---")
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'current_transcript' not in st.session_state:
        st.session_state.current_transcript = ""
    
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
                # create slightly smaller chunks for faster generation
                chunks = chunk_transcript(segments_for_summarizer, max_chars=1600)
                # use BART summarizer
                summaries = summarize_chunks_bart(chunks, model_name="facebook/bart-large-cnn", device=-1)
                merged = merge_summaries_text(summaries)
                # produce a global, more coherent summary
                final_summary = summarize_global(merged, model_name="facebook/bart-large-cnn", device=-1)
                final_summary = _sanitize(final_summary)
                # sanitize full text for metadata parsing/display
                full_text = _sanitize(full_text)
                structured = build_structure(segments_for_summarizer, final_summary, full_text)
                st.session_state.processed_data = structured
                st.session_state.current_transcript = full_text
                st.success("✅ Transcript processed successfully!")
                st.rerun()
        else:
            st.error("⚠️ Please provide a transcript or audio file first!")
    
    if st.sidebar.button("🗑️ Clear All", use_container_width=True):
        st.session_state.processed_data = None
        st.session_state.current_transcript = ""
        st.rerun()
    
    if st.session_state.processed_data:
        data = st.session_state.processed_data
        
        st.markdown("## 📋 Generated Meeting Minutes")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
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
        
        with col2:
            st.markdown("### 📊 Processing Stats")
            st.metric("Key Topics Extracted", len(data.get('key_topics', [])))
            st.metric("Decisions Identified", len(data.get('decisions', [])))
            st.metric("Action Items Found", len(data.get('action_items', [])))
            st.metric("Attendees Detected", len(data.get('attendees', [])))
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "👥 Attendees", 
            "🔑 Key Topics", 
            "📝 Summary", 
            "✅ Decisions", 
            "📌 Action Items", 
            "📅 Next Meeting"
        ])
        
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
        
        with tab2:
            st.markdown("### 🔑 Key Topics")
            key_topics = [ _sanitize(t) for t in data.get('key_topics', []) if t ]
            if key_topics:
                for i, topic in enumerate(key_topics):
                    data['key_topics'][i] = st.text_input(
                        f"Topic {i+1}",
                        value=topic,
                        key=f"topic_{i}"
                    )
                
                if st.button("➕ Add Topic"):
                    key_topics.append("")
                    st.rerun()
            else:
                st.info("No key topics extracted automatically.")
                if st.button("➕ Add Topic"):
                    data['key_topics'] = [""]
                    st.rerun()
        
        with tab3:
            st.markdown("### 📝 Discussion Summary")
            summary = _sanitize(data.get('summary', ''))
            bullets = _as_bullets(summary)
            if bullets:
                st.markdown("#### Key Points")
                st.markdown("\n".join([f"- {b}" for b in bullets]))
                st.markdown("---")
            data['summary'] = st.text_area(
                "Summary (editable)",
                value=summary,
                height=200,
                key="summary_edit",
                help="Edit the auto-generated summary"
            )
        
        with tab4:
            st.markdown("### ✅ Decisions")
            decisions = [ _sanitize(d) for d in data.get('decisions', []) ]
            if decisions:
                for i, decision in enumerate(decisions):
                    data['decisions'][i] = st.text_area(
                        f"Decision {i+1}",
                        value=decision,
                        height=80,
                        key=f"decision_{i}"
                    )
                
                if st.button("➕ Add Decision"):
                    decisions.append("")
                    st.rerun()
            else:
                st.info("No decisions detected. Click below to add manually.")
                if st.button("➕ Add Decision"):
                    data['decisions'] = [""]
                    st.rerun()
        
        with tab5:
            st.markdown("### 📌 Action Items")
            action_items = [
                {
                    'task': _sanitize(a.get('task','')),
                    'responsible': _sanitize(a.get('responsible','')),
                    'deadline': _sanitize(a.get('deadline','')),
                    'status': _sanitize(a.get('status','Pending')),
                }
                for a in data.get('action_items', [])
            ]
            if action_items:
                for i, action in enumerate(action_items):
                    st.markdown(f"**Action Item {i+1}**")
                    col1, col2, col3, col4 = st.columns([4, 2, 2, 1])
                    
                    with col1:
                        action_items[i]['task'] = st.text_area(
                            "Task",
                            value=action.get('task', ''),
                            key=f"task_{i}",
                            height=60,
                            help="Enter the action item task description"
                        )
                    with col2:
                        action_items[i]['responsible'] = st.text_input(
                            "Responsible",
                            value=action.get('responsible', ''),
                            key=f"responsible_{i}"
                        )
                    with col3:
                        action_items[i]['deadline'] = st.text_input(
                            "Deadline",
                            value=action.get('deadline', ''),
                            key=f"deadline_{i}"
                        )
                    with col4:
                        action_items[i]['status'] = st.selectbox(
                            "Status",
                            options=["Pending", "In progress", "Completed", "Upcoming"],
                            index=["Pending", "In progress", "Completed", "Upcoming"].index(
                                action.get('status', 'Pending')
                            ),
                            key=f"status_{i}"
                        )
                    st.markdown("---")
                
                if st.button("➕ Add Action Item"):
                    action_items.append({
                        'task': '',
                        'responsible': '',
                        'deadline': '',
                        'status': 'Pending'
                    })
                    st.rerun()
            else:
                st.info("No action items detected. Click below to add manually.")
                if st.button("➕ Add Action Item"):
                    data['action_items'] = [{
                        'task': '',
                        'responsible': '',
                        'deadline': '',
                        'status': 'Pending'
                    }]
                    st.rerun()
        
        with tab6:
            st.markdown("### 📅 Next Meeting")
            next_meeting = data.get('next_meeting', {})
            
            col1, col2 = st.columns(2)
            with col1:
                next_meeting['date'] = st.text_input(
                    "Date",
                    value=next_meeting.get('date') or "",
                    key="next_date"
                )
                next_meeting['time'] = st.text_input(
                    "Time",
                    value=next_meeting.get('time') or "",
                    key="next_time"
                )
            with col2:
                next_meeting['venue'] = st.text_input(
                    "Venue",
                    value=next_meeting.get('venue') or "",
                    key="next_venue"
                )
                next_meeting['agenda'] = st.text_area(
                    "Agenda",
                    value=next_meeting.get('agenda') or "",
                    key="next_agenda",
                    height=100
                )
        
        st.markdown("---")
        st.markdown("## 📥 Export Options")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📄 Export as PDF", type="primary", use_container_width=True):
                exporter = MeetingExporter()
                pdf_buffer = exporter.export_to_pdf(data)
                
                filename = f"Meeting_Minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_buffer,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📝 Export as DOCX", type="primary", use_container_width=True):
                exporter = MeetingExporter()
                docx_buffer = exporter.export_to_docx(data)
                
                filename = f"Meeting_Minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                st.download_button(
                    label="⬇️ Download DOCX",
                    data=docx_buffer,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
    
    else:
        st.info("👈 Please provide a meeting transcript using the sidebar and click 'Process Transcript' to generate structured minutes.")

        
        
        with st.expander("ℹ️ How to use AIMS"):
            st.markdown("""
            1. Select an input method in the sidebar (Paste Text / Upload PDF / Upload Text / Upload Audio).
            2. Provide the transcript or upload the audio file.
            3. Click "Process Transcript" to generate minutes.
            4. Review and edit the extracted information on the page.
            5. Export the result as PDF or DOCX using the buttons below.
            """)

if __name__ == "__main__":
    main()
