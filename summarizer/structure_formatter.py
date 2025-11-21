import re
from datetime import datetime
from typing import Dict
from pathlib import Path

# try to import optional HF punctuation and summarization tools
try:
    from transformers import pipeline
    HAVE_HF = True
except Exception:
    HAVE_HF = False

def extract_metadata_from_text(full_text):
    meta = {}
    # simple regex extractions
    title = re.search(r"Title:\s*(.+)", full_text, re.IGNORECASE)
    date = re.search(r"Date:\s*(.+)", full_text, re.IGNORECASE)
    time = re.search(r"Time:\s*(.+)", full_text, re.IGNORECASE)
    venue = re.search(r"Venue:\s*(.+)", full_text, re.IGNORECASE)
    organizer = re.search(r"Organizer:\s*(.+)", full_text, re.IGNORECASE)
    recorder = re.search(r"Recorder:\s*(.+)", full_text, re.IGNORECASE)
    meta["title"] = title.group(1).strip() if title else f"Meeting Summary {datetime.now().strftime('%Y-%m-%d')}"
    meta["date"] = date.group(1).strip() if date else datetime.now().strftime('%d/%m/%Y')
    meta["time"] = time.group(1).strip() if time else ""
    meta["venue"] = venue.group(1).strip() if venue else ""
    meta["organizer"] = organizer.group(1).strip() if organizer else ""
    meta["recorder"] = recorder.group(1).strip() if recorder else ""
    return meta

def extract_attendees(diarized_segments):
    """
    Extract attendees from diarized segments.
    Collects unique speaker names from segments and also looks for explicit attendee lists.
    """
    attendees = []
    seen_names = set()
    
    # First, collect unique speakers from segments
    if diarized_segments:
        for seg in diarized_segments:
            speaker = seg.get("speaker", "")
            if speaker and speaker not in seen_names:
                # Check if it's a real name (not just "Speaker 1", "Speaker 2")
                if not re.match(r'^Speaker\s+\d+$', speaker, re.IGNORECASE):
                    attendees.append({"name": speaker, "role": ""})
                    seen_names.add(speaker)
    
    # Also look for explicit "Attendees:" section in text
    text = " ".join([s.get("text", "") for s in diarized_segments[:20]])
    m = re.search(r"Attendees?:\s*(.+?)(?:\n\n|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        parts = re.split(r"[,\n;]+", m.group(1))
        for p in parts:
            p = p.strip()
            if p and len(p) > 2:
                if " - " in p or "–" in p or "—" in p:
                    name_role = re.split(r"\s*[-–—]\s*", p, maxsplit=1)
                    name = name_role[0].strip()
                    role = name_role[1].strip() if len(name_role) > 1 else ""
                    if name and name not in seen_names:
                        attendees.append({"name": name, "role": role})
                        seen_names.add(name)
                else:
                    if p not in seen_names:
                        attendees.append({"name": p, "role": ""})
                        seen_names.add(p)
    
    return attendees[:20]  # Limit to 20 attendees

def extract_decisions_and_actions(full_text):
    decisions = []
    actions = []
    seen_decisions = set()
    seen_actions = set()

    for line in full_text.splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue

        if re.search(r"\b(decid|decided|decision|agreed|approved|concluded|resolved)\b", line, re.IGNORECASE):
            if line not in seen_decisions:
                decisions.append(line)
                seen_decisions.add(line)

        if re.search(r"\b(action item|action|todo|to do|task|assign|assigned|will)\b", line, re.IGNORECASE) or re.match(r".+-\s*[A-Z][a-z]+", line):
            clean_line = re.sub(r"^[\-\*\•\s]+", "", line)
            split_pattern = r"\s[-–—]\s"
            if re.search(split_pattern, clean_line):
                raw_parts = re.split(split_pattern, clean_line)
            else:
                raw_parts = [clean_line]
            parts = [p.strip(" -–—") for p in raw_parts if p and p.strip(" -–—")]
            if not parts:
                continue

            # normalize generic prefixes before generating keys
            task_candidate = re.sub(
                r"^(action items?|action item|action|task|tasks?|todo|to-?do)\s*[:\-]\s*",
                "",
                parts[0],
                flags=re.IGNORECASE,
            ).strip()

            task_key = (task_candidate or parts[0])[:80].lower()
            if task_key in seen_actions:
                continue

            responsible = parts[1] if len(parts) > 1 else ""
            deadline = parts[2] if len(parts) > 2 else ""
            status = parts[3] if len(parts) > 3 else "Pending"

            # If the first segment is just a label (e.g., "Action item"), shift the window forward
            if len(parts) > 1 and re.fullmatch(r"(action items?|action|task|tasks?|todo|to-?do)\b\.?", parts[0], flags=re.IGNORECASE):
                task_candidate = parts[1]
                responsible = parts[2] if len(parts) > 2 else ""
                deadline = parts[3] if len(parts) > 3 else ""
                status = parts[4] if len(parts) > 4 else status

            task = re.sub(
                r"^(action items?|action item|action|task|tasks?|todo|to-?do)\s*[:\-]\s*",
                "",
                task_candidate,
                flags=re.IGNORECASE,
            ).strip() or task_candidate.strip()

            def _strip_label(value: str, pattern: str) -> str:
                if not value:
                    return value
                cleaned = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
                return cleaned

            responsible = _strip_label(responsible, r"^(responsible|owner|assignee|assigned to|lead)\s*[:\-]\s*")
            deadline = _strip_label(deadline, r"^(deadline|due date?|target date?|by)\s*[:\-]\s*")
            status = _strip_label(status, r"^(status)\s*[:\-]\s*") or "Pending"

            actions.append({
                "task": task or clean_line,
                "responsible": responsible,
                "deadline": deadline,
                "status": status
            })
            seen_actions.add(task_key)

    return decisions[:15], actions[:15]

def build_structure(diarized_segments, merged_summary, full_transcript_text):
    from nlp_processor import MeetingNLPProcessor

    processor = MeetingNLPProcessor()

    metadata = extract_metadata_from_text(full_transcript_text)
    attendees = extract_attendees(diarized_segments)
    decisions, actions = extract_decisions_and_actions(full_transcript_text + "\n" + merged_summary)

    # extract key topics (used as agenda titles)
    key_topics = processor.extract_key_topics(merged_summary)

    # build structured agenda
    agenda = []
    for idx, topic in enumerate(key_topics):
        agenda.append({
            "no": idx + 1,
            "title": topic,
            "discussion": "",         # user fills in UI
            "responsibility": ""      # user fills in UI
        })

    keywords = processor.extract_keywords(full_transcript_text, top_n=12)
    entity_actions = processor.extract_entity_actions(full_transcript_text)

    structured = {
        "metadata": metadata,
        "attendees": attendees,
        "agenda": agenda,                   # UPDATED
        "summary": merged_summary,
        "decisions": decisions,
        "action_items": actions,
        "next_meeting": {},
    }

    return structured


def _simple_cleanup(text: str) -> str:
    # basic normalizations: collapse whitespace, fix repeated chars/words, basic casing
    t = re.sub(r"\s+", " ", text).strip()
    # remove repeated short tokens like "uh uh" or "mmm mmm"
    t = re.sub(r"\b([a-z]{1,3})\s+\1\b", r"\1", t, flags=re.IGNORECASE)
    # remove multiple repeated punctuation
    t = re.sub(r"[!?]{2,}", ".", t)
    # ensure sentences start uppercase (naive)
    sentences = re.split(r'(?<=[\.\?\!])\s+', t)
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    return " ".join(sentences)

def _punctuate_text(text: str):
    """
    Try to restore punctuation using a HF model (if installed). Falls back to basic cleanup.
    Recommended models: 'oliverguhr/fullstop-punctuation-multilingual' or
    'kredor/punctuator' (may require extra deps).
    """
    if not HAVE_HF:
        return _simple_cleanup(text)

    try:
        # choose a lightweight punctuation model if available
        model_name = "oliverguhr/fullstop-punctuation-multilingual"
        punct = pipeline("text2text-generation", model=model_name, device=-1)
        # run in chunks to avoid very long inputs
        max_chunk = 3000
        out_parts = []
        for i in range(0, len(text), max_chunk):
            chunk = text[i : i + max_chunk]
            res = punct(chunk, max_length=chunk and min(1024, len(chunk) + 50))
            if isinstance(res, list):
                out_parts.append(res[0].get("generated_text", chunk))
            else:
                out_parts.append(str(res))
        punctuated = " ".join(out_parts)
        return _simple_cleanup(punctuated)
    except Exception:
        return _simple_cleanup(text)

def generate_structured_minutes(
    full_transcript_text: str,
    diarized_segments=None,
    model_name: str = "sshleifer/distilbart-cnn-12-6",
    device: int = None,
) -> Dict:
    """
    Generates structured minutes (Title, Date, Attendees, Agenda, Discussion Points,
    Decisions, Action Items) using a small summarization model + postprocessing.
    Returns a dict ready for the UI.
    """
    # 1) Punctuate & cleanup
    cleaned = _punctuate_text(full_transcript_text)

    # 2) Build instruction prompt for structured output
    prompt = (
        "You are a meeting minutes assistant. Given the meeting transcript below, "
        "produce a JSON object with keys: title, date, attendees (list), agenda, "
        "discussion (short bullets), decisions (list), action_items (list of {task,responsible,deadline}). "
        "Keep answers concise. Transcript:\n\n"
        f"{cleaned}\n\n"
        "Return ONLY a JSON object."
    )

    # 3) Try transformers summarizer/generator pipeline if available, else fallback simple extraction
    structured_text = ""
    if HAVE_HF:
        try:
            gen = pipeline("text2text-generation", model=model_name, device=device)
            # chunk prompt if too large; here we pass the full prompt (smaller models may truncate)
            out = gen(prompt, max_length=512, truncation=True)
            if isinstance(out, list) and out:
                structured_text = out[0].get("generated_text", "")
            else:
                structured_text = str(out)
        except Exception:
            structured_text = ""
    if not structured_text:
        # fallback: naive extraction using regex and splitting
        metadata = {"title": "", "date": "", "attendees": [], "agenda": "", "discussion": "", "decisions": [], "action_items": []}
        # try to extract Title/Date blocks
        tmatch = re.search(r"Title[:\-]\s*(.+)", full_transcript_text, re.IGNORECASE)
        dmatch = re.search(r"Date[:\-]\s*([0-9/.-]+)", full_transcript_text, re.IGNORECASE)
        metadata["title"] = tmatch.group(1).strip() if tmatch else f"Meeting {datetime.now().strftime('%Y-%m-%d')}"
        metadata["date"] = dmatch.group(1).strip() if dmatch else datetime.now().strftime("%d/%m/%Y")
        # attendees: look for 'Attendees' block
        att = re.search(r"Attendees[:\n]\s*(.+?)(?:\n\n|\Z)", full_transcript_text, re.IGNORECASE | re.DOTALL)
        if att:
            parts = re.split(r"[\n,;]+", att.group(1))
            metadata["attendees"] = [p.strip() for p in parts if p.strip()]
        # decisions & actions from keywords
        decisions = []
        actions = []
        for line in cleaned.splitlines():
            if re.search(r"\b(decid|decision|we decided)\b", line, re.IGNORECASE):
                decisions.append(line.strip())
            if re.search(r"\b(action|todo|to do|task|assign)\b", line, re.IGNORECASE):
                # naive parsing "task - person - date"
                parts = [p.strip() for p in re.split(r"-|–", line) if p.strip()]
                if parts:
                    task = parts[0]
                    responsible = parts[1] if len(parts) > 1 else ""
                    deadline = parts[2] if len(parts) > 2 else ""
                    actions.append({"task": task, "responsible": responsible, "deadline": deadline})
        metadata["decisions"] = decisions
        metadata["action_items"] = actions
        metadata["agenda"] = ""  # keep blank for fallback
        metadata["discussion"] = cleaned[:800]  # first chunk as summary
        return metadata

    # 4) Parse model output (expect JSON). Try to load JSON from model output.
    try:
        import json
        # model might output text before/after JSON; extract first {...}
        m = re.search(r"\{.*\}", structured_text, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            # normalize fields
            parsed.setdefault("title", parsed.get("title", ""))
            parsed.setdefault("date", parsed.get("date", ""))
            parsed.setdefault("attendees", parsed.get("attendees", []))
            parsed.setdefault("agenda", parsed.get("agenda", ""))
            parsed.setdefault("discussion", parsed.get("discussion", ""))
            parsed.setdefault("decisions", parsed.get("decisions", []))
            parsed.setdefault("action_items", parsed.get("action_items", []))
            return parsed
    except Exception:
        pass

    # last-resort fallback: return minimal structure
    return {
        "title": "",
        "date": "",
        "attendees": [],
        "agenda": "",
        "discussion": cleaned[:800],
        "decisions": [],
        "action_items": [],
    }