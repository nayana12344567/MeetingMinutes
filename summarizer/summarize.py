# try heavy imports, but allow graceful fallback
try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False

# sentence-transformers optional (not required for the fallback)
try:
    from sentence_transformers import SentenceTransformer, util
    HAVE_ST = True
except Exception:
    HAVE_ST = False

import math
from typing import List, Dict

def chunk_transcript(segments: List[Dict], max_chars: int = 3000) -> List[Dict]:
    """
    Group segments into chunks ~max_chars by concatenating consecutive segments.
    Returns list of chunk dicts {'start','end','text','segments'}.
    """
    chunks = []
    cur = {"start": None, "end": None, "text": "", "segments": []}
    for s in segments:
        text = s.get("text", "").strip()
        if not text:
            continue
        if cur["start"] is None:
            cur["start"] = s.get("start", 0)
        cur["end"] = s.get("end", s.get("start", 0))
        speaker = s.get("speaker", "")
        entry = f"{speaker}: {text}" if speaker else text
        if cur["text"] and (len(cur["text"]) + len(entry) > max_chars):
            chunks.append(cur)
            cur = {"start": s.get("start", 0), "end": s.get("end", 0), "text": entry, "segments": [s]}
        else:
            cur["text"] = (cur["text"] + "\n" + entry) if cur["text"] else entry
            cur["segments"].append(s)
    if cur["text"]:
        chunks.append(cur)
    return chunks

def _get_device():
    # returns device arg for transformers pipeline: 0+ for GPU, -1 for CPU
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1

def summarize_chunks(chunks: List[Dict], model_name: str = "sshleifer/distilbart-cnn-12-6", device: int = None) -> List[Dict]:
    """
    Summarize each chunk. Default uses a smaller distilBART model for speed.
    Use transformers pipeline if available, otherwise a simple fallback.
    Returns list of summaries.
    """
    summaries = []
    if device is None:
        device = _get_device()

    summarizer = None
    if HAVE_TRANSFORMERS:
        try:
            summarizer = pipeline("summarization", model=model_name, device=device)
        except Exception:
            summarizer = None

    for c in chunks:
        text = c.get("text", "")
        summary_text = None

        if summarizer:
            try:
                out = summarizer(text, max_length=160, min_length=30, truncation=True)
                if isinstance(out, list) and len(out) > 0 and "summary_text" in out[0]:
                    summary_text = out[0]["summary_text"]
            except Exception:
                summary_text = None

        if not summary_text:
            # naive fallback: take first 3 sentences
            s = text.replace("\n", " ").strip()
            parts = [p.strip() for p in s.split(".") if p.strip()]
            if parts:
                summary_text = ". ".join(parts[:3]) + (". " if parts[:3] else "")
            else:
                summary_text = s[:400]

        summaries.append({"start": c.get("start", 0), "end": c.get("end", 0), "summary": summary_text})
    return summaries

def merge_summaries(summaries: List[Dict]) -> str:
    return "\n\n".join([s["summary"] for s in summaries])