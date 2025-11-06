import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import spacy
from datetime import datetime

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

class MeetingNLPProcessor:
    def __init__(self):
        self.nlp = None
        self._load_spacy()
        
        self.filler_words = {
            'um', 'uh', 'hmm', 'like', 'you know', 'i mean', 'sort of', 
            'kind of', 'basically', 'actually', 'literally', 'so', 'well',
            'okay', 'ok', 'right', 'yeah', 'yes', 'no', 'maybe'
        }
        
        self.decision_patterns = [
            r"(we|they|team|everyone|all)\s+(decided|agreed|approved|concluded|resolved|determined|will)",
            r"(it was|has been)\s+(decided|agreed|approved|concluded|resolved)",
            r"(decision|agreement|approval|resolution)\s+(was|is|has been)\s+(made|reached)",
            r"(final|final decision|consensus)\s+(is|was|reached)",
            r"(?:let'?s|let us)\s+(?:finalize|confirm|agree on)",
            r"(?:agreed|approved|accepted|endorsed|ratified|confirmed)",
            r"(?:decision|agreed)[:,\s]",
        ]
        
        self.action_patterns = [
            r"action:\s*([A-Z][a-z]+)\s*-\s*(.+?)\s*-\s*deadline:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})",
            r"(\w+)\s+(will|should|must|needs to|has to|is to|I'?ll)\s+(.+?)(?:by|before|on|until|deadline:)\s+([^\n.]+)",
            r"(\w+)\s+(?:is responsible for|will handle|assigned to|tasked with)\s+(.+?)(?:by|before|on)?\s*([A-Z][a-z]+\s+\d+|[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}|\d+\s+[A-Z][a-z]+)?",
            r"(action item|task|to-do):\s*(.+?)(?:-|–)?\s*(?:assigned to|owner:)?\s*(\w+)(?:\s+by\s+([^.]+))?",
        ]
        
    def _load_spacy(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")
    
    def preprocess_text(self, text):
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
        text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
        text = re.sub(r'\[Speaker \d+\]:', '', text)
        
        for filler in self.filler_words:
            text = re.sub(r'\b' + re.escape(filler) + r'\b', '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        
        text = text.strip()
        return text
    
    def extract_metadata(self, text):
        metadata = {
            'title': None,
            'date': None,
            'time': None,
            'venue': None,
            'organizer': None,
            'recorder': None
        }
        
        title_match = re.search(r'(?:title|meeting|subject):\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        date_patterns = [
            r'(?:date|on):\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'(?:date|on):\s*(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                metadata['date'] = date_match.group(1)
                break
        
        time_match = re.search(r'(?:time|at):\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?(?:\s*[-–]\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)?)', text, re.IGNORECASE)
        if time_match:
            metadata['time'] = time_match.group(1).strip()
        
        venue_match = re.search(r'(?:venue|location|place):\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if venue_match:
            metadata['venue'] = venue_match.group(1).strip()
        
        organizer_match = re.search(r'(?:organizer|organized by):\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if organizer_match:
            metadata['organizer'] = organizer_match.group(1).strip()
        
        recorder_match = re.search(r'(?:recorder|recorded by|minutes by):\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if recorder_match:
            metadata['recorder'] = recorder_match.group(1).strip()
        
        return metadata
    
    def extract_attendees(self, text):
        attendees = []
        seen_names = set()
        
        attendees_section = re.search(r'(?:attendees|participants|present):\s*(.+?)(?:\n\n|---)', text, re.IGNORECASE | re.DOTALL)
        if attendees_section:
            section_text = attendees_section.group(1)
            lines = section_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = re.split(r'[-–—]', line, maxsplit=1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    role = parts[1].strip()
                    if name not in seen_names:
                        attendees.append({'name': name, 'role': role})
                        seen_names.add(name)
                else:
                    name_match = re.match(r'^([A-Z][a-zA-Z\s\.]+)', line)
                    if name_match:
                        name = name_match.group(1).strip()
                        if name not in seen_names:
                            attendees.append({'name': name, 'role': ''})
                            seen_names.add(name)
        
        intro_patterns = [
            r'(?:I am|I\'m|This is|My name is)\s+([A-Z][a-z]+)',
            r'([A-Z][a-z]+),\s+(?:Project coordinator|Technical head|Finance head|Sponsorship|Publicity|Marketing|Logistics)',
        ]
        
        for pattern in intro_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if name not in seen_names and len(name) > 2:
                    role_match = re.search(rf'{name},?\s+([^.\n]+)', text, re.IGNORECASE)
                    role = role_match.group(1).strip() if role_match else ''
                    attendees.append({'name': name, 'role': role})
                    seen_names.add(name)
        
        if not attendees:
            doc = self.nlp(text)
            common_non_names = {'technova', 'whisper', 'ai', 'college', 'meeting', 'fest', 'event'}
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    if len(name) > 2 and name.lower() not in common_non_names and name not in seen_names:
                        attendees.append({'name': name, 'role': ''})
                        seen_names.add(name)
        
        return attendees[:20]
    
    def extract_key_topics(self, text, top_n=5):
        sentences = sent_tokenize(text)
        
        if len(sentences) < 3:
            return []
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-top_n:][::-1]
            
            topics = [feature_names[i] for i in top_indices]
            return topics
        except:
            return []
    
    def generate_summary(self, text, num_sentences=3):
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            summary = ' '.join([sentences[i] for i in top_indices])
            return summary
        except:
            return ' '.join(sentences[:num_sentences])
    
    def extract_decisions(self, text):
        decisions = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            for pattern in self.decision_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    cleaned = sentence.strip()
                    if cleaned and cleaned not in decisions:
                        decisions.append(cleaned)
                    break
        
        return decisions
    
    def extract_action_items(self, text):
        action_items = []
        
        action_lines = re.finditer(r'action:\s*([A-Z][a-z]+)\s*-\s*(.+?)\s*-\s*deadline:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})', text, re.IGNORECASE)
        for match in action_lines:
            responsible = match.group(1).strip()
            task = match.group(2).strip()
            deadline = match.group(3).strip()
            
            action_items.append({
                'task': task,
                'responsible': responsible.capitalize(),
                'deadline': deadline,
                'status': 'Pending'
            })
        
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if sentence.lower().startswith('action:'):
                continue
                
            for pattern in self.action_patterns[1:]:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    responsible = groups[0] if groups[0] else "Not specified"
                    task = sentence.strip()
                    deadline = "Not specified"
                    
                    if len(groups) >= 2 and groups[1] not in ['will', 'should', 'must', 'needs to', 'has to', 'is to', "I'll", "i'll"]:
                        task_part = groups[2] if len(groups) > 2 else groups[1]
                        if task_part and len(task_part) > 3:
                            task = task_part.strip()
                    
                    for group in groups:
                        if group and re.search(r'\d', group):
                            deadline = group.strip()
                            break
                    
                    action_items.append({
                        'task': task,
                        'responsible': responsible.capitalize(),
                        'deadline': deadline,
                        'status': 'Pending'
                    })
                    break
        
        action_section = re.search(r'(?:action items?|tasks?|to-?do):\s*(.+?)(?:\n\n|---|\Z)', text, re.IGNORECASE | re.DOTALL)
        if action_section:
            section_text = action_section.group(1)
            lines = section_text.strip().split('\n')
            
            for line in lines:
                if not line.strip() or 'task' in line.lower() and 'responsible' in line.lower():
                    continue
                
                parts = re.split(r'\t+|\s{2,}', line.strip())
                if len(parts) >= 2:
                    task = parts[0].strip()
                    responsible = parts[1].strip() if len(parts) > 1 else "Not specified"
                    deadline = parts[2].strip() if len(parts) > 2 else "Not specified"
                    status = parts[3].strip() if len(parts) > 3 else "Pending"
                    
                    if task and len(task) > 5:
                        action_items.append({
                            'task': task,
                            'responsible': responsible,
                            'deadline': deadline,
                            'status': status
                        })
        
        seen = set()
        unique_actions = []
        for action in action_items:
            task_key = action['task'][:50].lower()
            if task_key not in seen:
                seen.add(task_key)
                unique_actions.append(action)
        
        return unique_actions[:15]
    
    def extract_next_meeting(self, text):
        next_meeting = {
            'date': None,
            'time': None,
            'venue': None,
            'agenda': None
        }
        
        next_section = re.search(r'(?:next meeting|upcoming meeting|follow-up):\s*(.+?)(?:\n\n|---|\Z)', text, re.IGNORECASE | re.DOTALL)
        if next_section:
            section_text = next_section.group(1)
            
            date_match = re.search(r'(?:date|on):\s*([^\n]+)', section_text, re.IGNORECASE)
            if date_match:
                next_meeting['date'] = date_match.group(1).strip()
            
            time_match = re.search(r'(?:time|at):\s*([^\n]+)', section_text, re.IGNORECASE)
            if time_match:
                next_meeting['time'] = time_match.group(1).strip()
            
            venue_match = re.search(r'(?:venue|location):\s*([^\n]+)', section_text, re.IGNORECASE)
            if venue_match:
                next_meeting['venue'] = venue_match.group(1).strip()
            
            agenda_match = re.search(r'(?:agenda|topics?):\s*([^\n]+)', section_text, re.IGNORECASE)
            if agenda_match:
                next_meeting['agenda'] = agenda_match.group(1).strip()
        
        return next_meeting
    
    def process_transcript(self, raw_text):
        cleaned_text = self.preprocess_text(raw_text)
        
        metadata = self.extract_metadata(raw_text)
        attendees = self.extract_attendees(raw_text)
        
        discussion_text = cleaned_text
        for section_name in ['attendees', 'action items', 'next meeting', 'decisions']:
            discussion_text = re.sub(
                rf'{section_name}:.*?(?=\n\n|---|\Z)',
                '',
                discussion_text,
                flags=re.IGNORECASE | re.DOTALL
            )
        
        key_topics = self.extract_key_topics(discussion_text, top_n=5)
        summary = self.generate_summary(discussion_text, num_sentences=3)
        decisions = self.extract_decisions(raw_text)
        action_items = self.extract_action_items(raw_text)
        next_meeting = self.extract_next_meeting(raw_text)
        
        return {
            'metadata': metadata,
            'attendees': attendees,
            'key_topics': key_topics,
            'summary': summary,
            'decisions': decisions,
            'action_items': action_items,
            'next_meeting': next_meeting,
            'original_text': raw_text,
            'cleaned_text': cleaned_text
        }
