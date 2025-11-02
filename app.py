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
from summarizer.summarize import chunk_transcript, summarize_chunks, merge_summaries
from summarizer.structure_formatter import build_structure

st.set_page_config(
    page_title="AIMS - AI Meeting Summarizer",
    page_icon="📝",
    layout="wide"
)

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
                    diarized = diarize_audio(audio_path, segments, out_json=diarize_json, use_pyannote=True)
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
            with st.spinner("Processing transcript using rule-based NLP and summarization..."):
                # prefer diarized segments if available
                if diarized:
                    segments_for_summarizer = diarized
                    full_text = transcript_text
                else:
                    # fall back to existing plain-text path: create simple segments
                    full_text = transcript_text
                    segments_for_summarizer = [{"speaker":"Speaker 1","start":0,"end":0,"text":full_text}]
                chunks = chunk_transcript(segments_for_summarizer, max_chars=3000)
                summaries = summarize_chunks(chunks)
                merged = merge_summaries(summaries)
                structured = build_structure(segments_for_summarizer, merged, full_text)
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
            key_topics = data.get('key_topics', [])
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
            summary = data.get('summary', '')
            data['summary'] = st.text_area(
                "Summary",
                value=summary,
                height=200,
                key="summary_edit",
                help="Edit the auto-generated summary"
            )
        
        with tab4:
            st.markdown("### ✅ Decisions")
            decisions = data.get('decisions', [])
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
            action_items = data.get('action_items', [])
            if action_items:
                for i, action in enumerate(action_items):
                    st.markdown(f"**Action Item {i+1}**")
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        action_items[i]['task'] = st.text_input(
                            "Task",
                            value=action.get('task', ''),
                            key=f"task_{i}"
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

        
        with st.expander("📖 Sample Transcript Format"):
            st.code("""
[00:00:02] Sakshi: Good afternoon everyone. Let's start the meeting.
[00:00:08] Nayana: Good afternoon. I have the draft agenda ready.
[00:00:12] Prathiksha: Hi — I joined a bit late, sorry.
[00:00:20] Nikitha: No problem. Sakshi, could you recap the goal for today's meeting?
[00:00:25] Sakshi: Sure — we're planning the College Technical Fest. Today we'll discuss proposed events, sponsorship outreach, and the initial budget.
[00:01:05] Nayana: For events, I'm proposing a 24-hour hackathon, web design challenge, and a robotics line-follower contest.
[00:02:18] Prathiksha: Budget-wise, workshops will need ~₹5,000 for materials.
[00:03:10] Nikitha: I'll handle sponsorship outreach. I'll prepare a sponsor list and email template.
[00:03:45] Sakshi: Action: Nikitha to approach sponsors; deadline 31/10/2025.
[00:04:05] Prathiksha: Action: Allocate ₹5,000 for workshop materials — Sakshi to approve by 30/10/2025.
[00:04:40] Nayana: Next meeting scheduled for 02/11/2025 at 3:00 PM in Project Lab.
[00:04:50] Sakshi: Thanks everyone. Meeting adjourned.
    """, language="text")
        
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
