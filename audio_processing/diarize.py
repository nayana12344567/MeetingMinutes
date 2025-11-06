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
        # 2) Otherwise, alternate speakers when there are long gaps or keep Speaker 1
        last_speaker = "Speaker 1"
        for i, seg in enumerate(transcript_segments):
            txt = (seg.get("text") or "").strip()
            sp = last_speaker
            # pattern: Name: content
            try:
                import re
                m = re.match(r"^([A-Z][A-Za-z\.\- ]{1,30}):\s+(.*)$", txt)
                if m:
                    sp = m.group(1).strip()
                    txt = m.group(2).strip()
            except Exception:
                pass
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