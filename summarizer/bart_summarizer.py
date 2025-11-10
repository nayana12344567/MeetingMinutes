try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False

from typing import List, Dict

# Cache summarization pipelines per model to avoid reloading
_PIPELINE_CACHE: Dict[str, object] = {}

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

def summarize_global(text: str, model_name: str = "facebook/bart-large-cnn", device: int = -1) -> str:
    summarizer = get_bart_summarizer(model_name=model_name, device=device)
    if not summarizer:
        return text
    try:
        max_len = _calculate_max_length(text, summarizer)
        min_len = _calculate_min_length(text, summarizer, max_len)
        out = summarizer(text, max_length=max_len, min_length=min_len, truncation=True, do_sample=False, num_beams=4)
        if isinstance(out, list) and out and "summary_text" in out[0]:
            return out[0]["summary_text"]
        return text
    except Exception:
        return text


