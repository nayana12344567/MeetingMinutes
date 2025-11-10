import json
from pathlib import Path

def _save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

def diarize_audio(audio_path, transcript_segments, out_json=None, use_pyannote=True):
    """
    Attempt speaker diarization and align with transcript_segments.
    Returns list of segments with 'speaker','start','end','text'.
    Fallback: single speaker for all segments.
    """
    diarized = []
    try:
        if use_pyannote:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
            diarization = pipeline(audio_path)
            # convert to list of (start,end, speaker_label)
            turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
            # naive alignment: assign each transcript segment to the speaker whose turn overlaps midpoint
            for seg in transcript_segments:
                mid = (seg["start"] + seg["end"]) / 2.0
                sp = "Speaker 1"
                for t in turns:
                    if t["start"] <= mid <= t["end"]:
                        sp = t["speaker"]
                        break
                diarized.append({
                    "speaker": sp,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                })
        else:
            raise Exception("pyannote disabled")
    except Exception:
        # lightweight heuristic fallback:
        # 1) If segment text starts with "Name: ...", use that as speaker and strip prefix
        # 2) Extract speaker names from transcript patterns
        # 3) Track unique speakers and assign them properly
        import re
        speakers_seen = {}  # Map speaker name to speaker label
        speaker_counter = 1
        last_speaker = None
        
        for i, seg in enumerate(transcript_segments):
            txt = (seg.get("text") or "").strip()
            sp = None
            
            # Pattern 1: "Name: content" at start of text
            m = re.match(r"^([A-Z][A-Za-z\.\- ]{1,30}):\s+(.*)$", txt)
            if m:
                sp_name = m.group(1).strip()
                txt = m.group(2).strip()
                # Normalize speaker name (remove common prefixes/suffixes)
                sp_name = re.sub(r'\s+', ' ', sp_name)
                if sp_name not in speakers_seen:
                    speakers_seen[sp_name] = f"Speaker {speaker_counter}"
                    speaker_counter += 1
                sp = speakers_seen[sp_name]
            else:
                # Pattern 2: Look for speaker labels in brackets like [Speaker 1] or [Name]
                bracket_match = re.search(r'\[(?:Speaker\s+)?([A-Z][A-Za-z\.\- ]{1,30}|Speaker\s+\d+)\]', txt)
                if bracket_match:
                    sp_name = bracket_match.group(1).strip()
                    txt = re.sub(r'\[(?:Speaker\s+)?[A-Z][A-Za-z\.\- ]{1,30}|Speaker\s+\d+\]', '', txt).strip()
                    if sp_name not in speakers_seen:
                        speakers_seen[sp_name] = f"Speaker {speaker_counter}"
                        speaker_counter += 1
                    sp = speakers_seen[sp_name]
                # Pattern 3: Look for "SPEAKER_NAME:" pattern anywhere in text
                elif ':' in txt:
                    colon_parts = txt.split(':', 1)
                    potential_name = colon_parts[0].strip()
                    if len(potential_name) > 2 and len(potential_name) < 40 and re.match(r'^[A-Z][A-Za-z\.\- ]+$', potential_name):
                        sp_name = potential_name
                        txt = colon_parts[1].strip() if len(colon_parts) > 1 else txt
                        if sp_name not in speakers_seen:
                            speakers_seen[sp_name] = f"Speaker {speaker_counter}"
                            speaker_counter += 1
                        sp = speakers_seen[sp_name]
            
            # If no speaker found, use last speaker or default to Speaker 1
            if not sp:
                sp = last_speaker if last_speaker else "Speaker 1"
            
            diarized.append({
                "speaker": sp,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": txt
            })
            last_speaker = sp

    if out_json:
        _save_json(diarized, out_json)
    return diarized