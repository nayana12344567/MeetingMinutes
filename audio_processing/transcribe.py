import json
import os
from pathlib import Path
import tempfile

def _save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

def transcribe_with_whisper(file_path, model_name="small", language=None, out_json=None):
    """
    Try whisperx/whisper. Returns dict with 'text' and 'segments' (start,end,text).
    Saves JSON if out_json provided.
    """
    try:
        import whisperx
        # whisperx provides improved alignment + diarization hooks
        model = whisperx.load_model(model_name, device="cpu")
        result = model.transcribe(file_path, language=language)
        # result has "segments"
        transcript = {
            "text": result.get("text", ""),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"]} for s in result.get("segments", [])
            ]
        }
    except Exception:
        try:
            import whisper
            model = whisper.load_model(model_name)
            result = model.transcribe(file_path, language=language)
            transcript = {
                "text": result.get("text", ""),
                "segments": [
                    {"start": s["start"], "end": s["end"], "text": s["text"]} for s in result.get("segments", [])
                ]
            }
        except Exception as e:
            transcript = {"text": "", "segments": []}
            transcript["error"] = str(e)

    if out_json:
        _save_json(transcript, out_json)
    return transcript

def transcribe_audio(uploaded_file, tmp_dir=None, model_name="small", language=None):
    """
    Save uploaded_file (Streamlit UploadedFile) to temp path and transcribe.
    Returns (audio_path, transcript_dict, json_path)
    """
    tmp_dir = tmp_dir or tempfile.gettempdir()
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".wav"
    tmp_path = os.path.join(tmp_dir, f"meeting_audio{suffix}")
    with open(tmp_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())

    json_path = os.path.join(tmp_dir, Path(uploaded_file.name).stem + "_transcript.json")
    transcript = transcribe_with_whisper(tmp_path, model_name=model_name, language=language, out_json=json_path)
    return tmp_path, transcript, json_path