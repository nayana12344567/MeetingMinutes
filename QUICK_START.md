# Quick Start Guide

## 🚀 Fastest Way to Get Started

### Option 1: Automated Setup (Recommended)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# 2. Run setup script
python setup.py

# 3. Run the app
streamlit run app.py --server.port 5000
```

### Option 2: Manual Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download language models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. Run the app
streamlit run app.py --server.port 5000
```

## 📋 What Gets Installed

### Required Packages:
- ✅ **streamlit** - Web framework
- ✅ **transformers** - BART summarization models
- ✅ **torch** - PyTorch (for ML models)
- ✅ **nltk, spacy** - NLP processing
- ✅ **scikit-learn** - Key topics extraction
- ✅ **pdfplumber, pypdf2** - PDF processing
- ✅ **python-docx, reportlab** - Export to DOCX/PDF

### Optional Packages (for audio features):
- ⚠️ **openai-whisper** - Audio transcription
- ⚠️ **pyannote.audio** - Speaker diarization (requires Hugging Face token)
- ⚠️ **sentence-transformers** - Advanced embeddings

## 🎯 Features Available

### Without Optional Packages:
- ✅ Text transcript processing
- ✅ PDF/Text file upload
- ✅ Summarization
- ✅ Key topics extraction
- ✅ Action items & decisions extraction
- ✅ Export to PDF/DOCX

### With Optional Packages:
- ✅ Audio file transcription (Whisper)
- ✅ Speaker diarization (pyannote.audio)
- ✅ Better speaker identification

## ⚡ Troubleshooting

### "Module not found" errors:
```bash
pip install -r requirements.txt
```

### spaCy model not found:
```bash
python -m spacy download en_core_web_sm
```

### NLTK data missing:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Transformers/PyTorch installation issues:
```bash
# For CPU-only systems
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

## 🔑 Hugging Face Setup (for pyannote.audio)

Only needed if you want advanced speaker diarization:

1. Sign up at https://huggingface.co/
2. Get token from https://huggingface.co/settings/tokens
3. Run: `huggingface-cli login`
4. Accept terms for:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

## 📝 Usage

1. Open browser to `http://localhost:5000`
2. Choose input method:
   - Paste Text
   - Upload PDF
   - Upload Text File
   - Upload Audio (requires whisper)
3. Click "Process Transcript"
4. Review and edit results
5. Export as PDF or DOCX

## 💡 Tips

- Start with text input to test the app
- Audio transcription takes time (use "tiny" model for testing)
- The app works without pyannote.audio (uses fallback diarization)
- All models download automatically on first use

