"""
Parse transcripts with timestamps and speaker names into structured format.
Handles various formats from Zoom, Whisper AI, and other transcription services.
"""
import re
from typing import List, Dict


def parse_transcript_with_timestamps(text: str) -> List[Dict]:
    """
    Parse a transcript text that contains timestamps and speaker names.
    
    Handles formats like:
    - [00:00:02] Speaker Name: text
    - [00:00:02] Speaker Name - text
    - 00:00:02 Speaker Name: text
    - [HH:MM:SS] Name: text
    - Name: text (without timestamps)
    
    Returns list of segments with 'speaker', 'start', 'end', 'text'
    """
    if not text or not text.strip():
        return []
    
    segments = []
    lines = text.split('\n')
    speakers_seen = {}  # Map speaker name to normalized label
    speaker_counter = 1
    last_speaker = None
    last_timestamp = 0.0
    
    # Pattern 1: [HH:MM:SS] or [MM:SS] or [H:MM:SS] Speaker Name: text
    pattern1 = re.compile(
        r'\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?\s*'  # timestamp [H]H:MM[:SS]
        r'([A-Z][A-Za-z\.\-\s]{1,40}?)[:\-]\s*'      # speaker name followed by : or -
        r'(.*)$',                                      # text
        re.MULTILINE
    )
    
    # Pattern 2: Speaker Name: text (no timestamp)
    pattern2 = re.compile(
        r'^([A-Z][A-Za-z\.\-\s]{1,40}?)[:\-]\s*'     # speaker name
        r'(.*)$',                                      # text
        re.MULTILINE
    )
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        timestamp_seconds = None
        speaker_name = None
        text_content = None
        
        # Try pattern 1 first (with timestamp)
        match1 = pattern1.match(line)
        if match1:
            hours = int(match1.group(1))
            minutes = int(match1.group(2))
            seconds = int(match1.group(3)) if match1.group(3) else 0
            
            # Handle MM:SS format (treat as minutes:seconds)
            if hours > 59:  # Likely MM:SS format
                timestamp_seconds = hours * 60 + minutes
            else:
                timestamp_seconds = hours * 3600 + minutes * 60 + seconds
            
            speaker_name = match1.group(4).strip()
            text_content = match1.group(5).strip()
        else:
            # Try pattern 2 (no timestamp)
            match2 = pattern2.match(line)
            if match2:
                speaker_name = match2.group(1).strip()
                text_content = match2.group(2).strip()
                # Use last timestamp + small increment
                timestamp_seconds = last_timestamp + 1.0
        
        # If we found speaker and text, process it
        if speaker_name and text_content:
            # Normalize speaker name
            speaker_name = re.sub(r'\s+', ' ', speaker_name).strip()
            
            # Skip if speaker name looks like a timestamp or is too generic
            if re.match(r'^\d+$', speaker_name) or len(speaker_name) < 2:
                continue
            
            # Map speaker to normalized label
            # Keep original speaker name if it looks like a real name, otherwise use Speaker N
            if speaker_name not in speakers_seen:
                # Check if it's already a generic "Speaker N" label
                if re.match(r'^Speaker\s+\d+$', speaker_name, re.IGNORECASE):
                    speakers_seen[speaker_name] = speaker_name  # Keep as is
                else:
                    # Use original name if it's a real name (not too long, has proper format)
                    if len(speaker_name) <= 50 and re.match(r'^[A-Z][A-Za-z\.\-\s]+$', speaker_name):
                        speakers_seen[speaker_name] = speaker_name  # Keep original name
                    else:
                        speakers_seen[speaker_name] = f"Speaker {speaker_counter}"
                        speaker_counter += 1
            
            normalized_speaker = speakers_seen[speaker_name]
            
            # Calculate end time (use next timestamp or estimate)
            end_time = timestamp_seconds + 5.0  # Default 5 seconds per segment
            
            segments.append({
                'speaker': normalized_speaker,
                'start': timestamp_seconds if timestamp_seconds is not None else last_timestamp,
                'end': end_time,
                'text': text_content
            })
            
            if timestamp_seconds is not None:
                last_timestamp = timestamp_seconds
            last_speaker = normalized_speaker
    
    # If no segments found with patterns, try to extract from plain text
    if not segments:
        # Look for any lines with "Name: text" pattern
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple pattern: Name: text
            colon_match = re.match(r'^([A-Z][A-Za-z\.\-\s]{2,40}?)[:\-]\s+(.+)$', line)
            if colon_match:
                speaker_name = colon_match.group(1).strip()
                text_content = colon_match.group(2).strip()
                
                if speaker_name not in speakers_seen:
                    speakers_seen[speaker_name] = f"Speaker {speaker_counter}"
                    speaker_counter += 1
                
                segments.append({
                    'speaker': speakers_seen[speaker_name],
                    'start': len(segments) * 5.0,  # Estimate timestamps
                    'end': (len(segments) + 1) * 5.0,
                    'text': text_content
                })
    
    return segments


def has_timestamp_format(text: str) -> bool:
    """
    Check if text appears to contain timestamp patterns.
    """
    if not text:
        return False
    
    # Check for common timestamp patterns
    timestamp_patterns = [
        r'\[\d{1,2}:\d{2}(?::\d{2})?\]',  # [HH:MM:SS] or [MM:SS]
        r'^\d{1,2}:\d{2}(?::\d{2})?\s+',  # HH:MM:SS at start of line
    ]
    
    for pattern in timestamp_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    
    return False

