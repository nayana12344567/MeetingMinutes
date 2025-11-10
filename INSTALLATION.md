# Installation Guide for MeetingMinutesAI

## Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- VS Code (or any Python IDE)

## Step 1: Create Virtual Environment

Open terminal in VS Code (Ctrl + `) and run:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
# source .venv/bin/activate
```

## Step 2: Install Core Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Install spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

## Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

## Step 5: (Optional) Setup Hugging Face Token for pyannote.audio

If you want to use pyannote.audio for speaker diarization:

1. Create account at https://huggingface.co/
2. Get your access token from https://huggingface.co/settings/tokens
3. Install huggingface_hub and login:

```bash
pip install huggingface_hub
huggingface-cli login
```

4. Accept the terms for pyannote models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

## Step 6: Run the Application

```bash
streamlit run app.py --server.port 5000
```

The app will open in your browser at `http://localhost:5000`

## Troubleshooting

### If transformers installation fails:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

### If pyannote.audio installation fails:
- It's optional, the app will work without it
- Speaker diarization will use fallback method

### If whisper installation fails:
- Try: `pip install openai-whisper`
- Or use whisperx: `pip install whisperx`

### Memory Issues:
- For CPU-only systems, use smaller models
- The app uses "tiny" model by default for faster processing
- You can change model in app.py line 114

## Minimum Installation (Without Optional Features)

If you only want text processing (no audio):

```bash
pip install streamlit nltk pdfplumber pypdf2 python-docx reportlab scikit-learn spacy transformers torch
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Verify Installation

Run this to check if all packages are installed:

```bash
python -c "import streamlit, transformers, torch, nltk, spacy; print('All core packages installed!')"
```

