import re
from typing import List, Dict

try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False

# Cache summarization pipelines per model to avoid reloading
_PIPELINE_CACHE: Dict[str, object] = {}
_INSTRUCTION_PHRASES = (
    "create a clear professional meeting summary",
    "format:",
    "transcript:",
    "speaker names or filler words",
)

def _calculate_max_length(text: str, summarizer) -> int:
    """
    Calculate appropriate max_length based on input text length.
    For summarization, max_length should be ~50-60% of input length.
    Returns a value between 50 and 500.
    """
    if not summarizer or not text:
        return 200  # default fallback
    
    try:
        # Try to get tokenizer from the pipeline
        tokenizer = summarizer.tokenizer
        if tokenizer:
            # Tokenize the input to get actual token count
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False, return_tensors=None)
            input_length = len(tokens)
        else:
            # Fallback: estimate tokens from characters (rough approximation: 1 token ≈ 4 chars)
            input_length = len(text) // 4
    except Exception:
        # Fallback: estimate tokens from characters
        input_length = len(text) // 4
    
    # Calculate max_length as 60-75% of input for better summaries, but within reasonable bounds
    # For very short inputs, use at least 60% but minimum 80 tokens
    # For longer inputs, cap at 600 tokens to allow more comprehensive summaries
    if input_length < 150:
        max_len = max(80, int(input_length * 0.65))
    elif input_length < 500:
        max_len = int(input_length * 0.7)
    elif input_length < 1000:
        max_len = int(input_length * 0.65)
    else:
        max_len = min(600, int(input_length * 0.6))
    
    return max(80, min(600, max_len))  # Ensure it's between 80 and 600

def _calculate_min_length(text: str, summarizer, max_length: int) -> int:
    """
    Calculate appropriate min_length based on max_length.
    Typically 25-35% of max_length, but with reasonable bounds.
    """
    min_len = max(40, int(max_length * 0.3))
    return min(150, min_len)  # Cap at 150 for better summaries


def _clean_transcript_for_global_summary(text: str) -> str:
    """
    Lightly clean transcript text before sending it to the global summarizer.
    Only strip obvious speaker tags (e.g., "Speaker 1:" or "Alicia:") that appear
    at the beginning of lines so that real content like "Budget: $5k" is preserved.
    """
    if not text:
        return ""

    cleaned = text
    # Remove explicit "Speaker 1:" type labels at the beginning of a line
    cleaned = re.sub(r"(?im)^\s*Speaker\s*\d*:?\s*", "", cleaned)
    # Remove simple name labels at the beginning of a line (1-3 words)
    cleaned = re.sub(
        r"(?im)^\s*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*:\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(oo|umm+|aa+|ok|yes|done|trending now|I'll finish|I will finish)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _looks_like_instruction(line: str) -> bool:
    lowered = line.lower()
    return any(phrase in lowered for phrase in _INSTRUCTION_PHRASES)


def _format_summary_output(summary_text: str) -> str:
    """
    Ensure the final summary follows the expected UI contract:
      - first section is a short paragraph (2-3 sentences)
      - followed by up to 8 bullet points
    If the model already produced bullets, keep them but normalize the prefix.
    """
    if not summary_text:
        return ""

    raw = summary_text.strip()
    if not raw:
        return ""

    lines = [
        l.strip() for l in raw.splitlines()
        if l.strip() and not _looks_like_instruction(l)
    ]
    if len(lines) >= 2 and any(l.startswith(("-", "•")) for l in lines[1:]):
        paragraph = lines[0]
        bullets = []
        for line in lines[1:]:
            if not line or _looks_like_instruction(line):
                continue
            cleaned_line = line.lstrip("-• ").strip()
            if not cleaned_line:
                continue
            bullets.append(f"- {cleaned_line}")
            if len(bullets) >= 8:
                break
        if bullets:
            return "\n".join([paragraph] + bullets)
        return paragraph

    # Otherwise derive paragraph + bullets from sentences.
    sentences = []
    for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", raw)):
        s = s.strip()
        if not s or _looks_like_instruction(s):
            continue
        sentences.append(s)
    if not sentences:
        return raw

    intro = " ".join(sentences[:2]).strip()
    bullet_candidates = sentences[2:]
    bullets = []
    for sent in bullet_candidates:
        cleaned_line = re.sub(r"^[\-\•\*\s]+", "", sent).strip()
        if len(cleaned_line.split()) < 4:
            continue
        bullets.append(f"- {cleaned_line}")
        if len(bullets) >= 8:
            break

    # If we still don't have bullets, fall back to shorter sentences to ensure structure.
    if not bullets:
        for sent in sentences[2:6]:
            cleaned_line = re.sub(r"^[\-\•\*\s]+", "", sent).strip()
            if not cleaned_line:
                continue
            bullets.append(f"- {cleaned_line}")
            if len(bullets) >= 4:
                break

    if bullets:
        return "\n".join([intro] + bullets).strip()
    return intro

def get_bart_summarizer(model_name: str = "sshleifer/distilbart-cnn-12-6", device: int = -1):
    if not HAVE_TRANSFORMERS:
        return None
    key = f"{model_name}:{device}"
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = pipeline("summarization", model=model_name, device=device)
    return _PIPELINE_CACHE[key]

def summarize_chunks_bart(chunks: List[Dict], model_name: str = "sshleifer/distilbart-cnn-12-6", device: int = -1) -> List[Dict]:
    summaries: List[Dict] = []
    summarizer = get_bart_summarizer(model_name=model_name, device=device)

    for c in chunks:
        text = c.get("text", "")
        if summarizer:
            try:
                max_len = _calculate_max_length(text, summarizer)
                min_len = _calculate_min_length(text, summarizer, max_len)
                out = summarizer(
                    text,
                    max_length=max_len,
                    min_length=min_len,
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

def summarize_global(text, model_name="sshleifer/distilbart-cnn-12-6", device=-1):
    if not text or len(text) < 40:
        return text or ""

    # CLEAN NOISY TRANSCRIPT (Zoom/filler/repetitive)
    cleaned = _clean_transcript_for_global_summary(text)
    cleaned = cleaned[:3500]  # shorten for model

    prompt = (
        "Create a clear professional meeting summary without speaker names or filler words.\n"
        "Format:\n"
        "1 paragraph (2-4 lines)\n"
        "Then 4-8 bullet points of the main decisions / insights / next steps.\n\n"
        "Transcript:\n" + cleaned
    )

    summarizer = get_bart_summarizer(model_name=model_name, device=device)
    if summarizer:
        try:
            out = summarizer(
                prompt,
                max_length=240,
                min_length=110,
                truncation=True,
                do_sample=False,
                num_beams=4,
            )
            if isinstance(out, list) and out and "summary_text" in out[0]:
                return _format_summary_output(out[0]["summary_text"].strip())
        except Exception:
            pass

    # fallback: trimmed cleaned text to keep UI populated, but still structured
    return _format_summary_output(cleaned[:600].strip())
