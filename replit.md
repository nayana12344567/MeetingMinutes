# AIMS - AI Meeting Summarizer

## Overview

AIMS (AI Meeting Summarizer) is a rule-based NLP application that automatically generates structured meeting minutes from conversational transcripts. The system is designed to process raw meeting audio transcripts (such as those from WhisperAI, Google Meet, Zoom, or other transcription services) and extract key information including metadata, attendees, decisions, action items, and discussion topics. It provides export capabilities to DOCX and PDF formats for easy distribution and archiving.

The application uses a Streamlit-based web interface for user interaction and leverages natural language processing libraries (NLTK, spaCy, scikit-learn) to perform text analysis and information extraction.

**Current Status (October 29, 2025):**
- ✅ Fully functional MVP with tested rule-based NLP processing
- ✅ Successfully extracts attendees, decisions, and action items from conversational text
- ✅ PDF and DOCX export working correctly
- ✅ Tested with real WhisperAI transcripts

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses **Streamlit** as the web framework, providing an interactive single-page application with minimal setup. This choice enables rapid development and deployment while offering a clean, user-friendly interface for uploading transcripts and viewing results.

**Key design decisions:**
- Session state management for persisting processed data across user interactions
- Sidebar-based input selection (paste text, upload PDF, or upload text file)
- Wide layout configuration for better content presentation
- Multi-format input support (text, PDF via pdfplumber/PyPDF2)

### Backend Architecture - NLP Processing

The core processing logic resides in `MeetingNLPProcessor` class, which implements a **rule-based NLP approach** rather than machine learning models.

**Key components:**
- **Pattern Matching**: Uses regular expressions to identify decisions and action items
- **Named Entity Recognition**: Leverages spaCy's pre-trained models for entity extraction
- **Text Summarization**: Implements TF-IDF vectorization (scikit-learn) for extracting key discussion points
- **Text Cleaning**: Removes filler words and normalizes transcript text

**Decision patterns capture:**
- Team agreements and approvals
- Consensus statements
- Decision-making verbs (decided, agreed, approved, etc.)

**Action item patterns extract:**
- Assignee names
- Task descriptions
- Deadlines and dates
- Responsibility assignments

**Rationale**: Rule-based approach chosen over ML models for:
- Predictable, transparent results
- No training data requirements
- Lower computational overhead
- Easier debugging and customization

### Document Export Architecture

The `MeetingExporter` class handles document generation in multiple formats:

**DOCX Export (python-docx):**
- Structured document with headers and formatting
- Metadata section (title, date, time, venue, organizer)
- Styled paragraphs with bold labels

**PDF Export (ReportLab):**
- Professional layouts with tables
- Custom styling and spacing
- Platypus framework for document assembly

**Design choice**: Supporting multiple export formats accommodates different organizational preferences and use cases (sharing, archiving, printing).

### Data Flow

1. **Input Processing**: Text extraction from various sources (direct paste, PDF, text files)
2. **NLP Analysis**: Pattern matching and entity extraction on cleaned text
3. **Data Structuring**: Organizing extracted information into meeting_data dictionary
4. **Export Generation**: Converting structured data to formatted documents
5. **Download Delivery**: BytesIO streams for browser downloads

## External Dependencies

### NLP Libraries
- **spaCy** (`en_core_web_sm` model): Named entity recognition and linguistic analysis
- **NLTK**: Sentence tokenization, stopwords, and text preprocessing
- **scikit-learn**: TF-IDF vectorization for text summarization

### Document Processing
- **pdfplumber**: Primary PDF text extraction library
- **PyPDF2**: Fallback PDF reader for compatibility
- **python-docx**: DOCX document generation
- **ReportLab**: PDF document generation with advanced formatting

### Web Framework
- **Streamlit**: Web application framework and UI components

### No Database Layer
The application operates **statelessly** using Streamlit's session state for temporary storage. No persistent database is implemented, making it suitable for single-session processing without data retention requirements.

**Rationale**: Meeting summaries are generated on-demand and immediately exported, eliminating the need for server-side storage. This simplifies deployment and reduces infrastructure requirements.

### File I/O
All file operations use **BytesIO** in-memory streams for handling uploads and downloads, avoiding local filesystem dependencies and enabling cloud deployment compatibility.