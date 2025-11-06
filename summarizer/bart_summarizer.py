try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False

from typing import List, Dict

# Cache summarization pipelines per model to avoid reloading
_PIPELINE_CACHE: Dict[str, object] = {}

def get_bart_summarizer(model_name: str = "facebook/bart-large-cnn", device: int = -1):
    if not HAVE_TRANSFORMERS:
        return None
    key = f"{model_name}:{device}"
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = pipeline("summarization", model=model_name, device=device)
    return _PIPELINE_CACHE[key]

def summarize_chunks_bart(chunks: List[Dict], model_name: str = "facebook/bart-large-cnn", device: int = -1) -> List[Dict]:
    summaries: List[Dict] = []
    summarizer = get_bart_summarizer(model_name=model_name, device=device)

    for c in chunks:
        text = c.get("text", "")
        if summarizer:
            try:
                out = summarizer(
                    text,
                    max_length=300,
                    min_length=80,
                    truncation=True,
                    do_sample=False,
                    num_beams=4,
                )
                if isinstance(out, list) and out and "summary_text" in out[0]:
                    summary_text = out[0]["summary_text"]
                else:
                    summary_text = text[:600]
            except Exception:
                summary_text = text[:600]
        else:
            # Fallback: return first 3 sentences
            s = text.replace("\n", " ").strip()
            parts = [p.strip() for p in s.split(".") if p.strip()]
            summary_text = ". ".join(parts[:3]) + ("." if parts[:3] else "")

        summaries.append({
            "start": c.get("start", 0),
            "end": c.get("end", 0),
            "summary": summary_text,
        })

    return summaries

def merge_summaries_text(summaries: List[Dict]) -> str:
    return "\n\n".join([s.get("summary", "") for s in summaries if s.get("summary")])

def summarize_global(text: str, model_name: str = "facebook/bart-large-cnn", device: int = -1) -> str:
    summarizer = get_bart_summarizer(model_name=model_name, device=device)
    if not summarizer:
        return text
    try:
        out = summarizer(text, max_length=420, min_length=120, truncation=True, do_sample=False, num_beams=4)
        if isinstance(out, list) and out and "summary_text" in out[0]:
            return out[0]["summary_text"]
        return text
    except Exception:
        return text


