from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import re
from datetime import datetime

class TextProcessor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with free sentence transformer model"""
        self.embedding_model = SentenceTransformer(model_name)
        self.date_patterns = [
            r'\b\d{1,2}-[A-Za-z]{3,9}\b',  # 6-Sept, 15-December
            r'\b[A-Za-z]{3,9}\s+\d{1,2}\b',  # Sept 6, December 15
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # 6/9/2024
            r'\b\d{4}-\d{2}-\d{2}\b'  # 2024-09-06
        ]
    
    def normalize_dates(self, text: str) -> str:
        """Normalize different date formats for better matching"""
        date_replacements = {}
        
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                normalized_dates = self.expand_date_formats(match)
                date_replacements[match] = f"{match} ({' | '.join(normalized_dates)})"
        
        for original, expanded in date_replacements.items():
            text = text.replace(original, expanded)
        
        return text
    
    def expand_date_formats(self, date_str: str) -> List[str]:
        """Expand a date into multiple formats"""
        formats = []
        
        if re.match(r'\d{1,2}-[A-Za-z]{3,9}', date_str):
            day, month = date_str.split('-')
            formats.extend([
                f"{month} {day}",
                f"{month} {day}th" if day.endswith('1') else f"{month} {day}th",
                f"September {day}" if month.lower().startswith('sep') else f"{month} {day}",
                f"{day} {month}",
                date_str
            ])
        
        return list(set(formats))
    
    def chunk_text_smart(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Smart text chunking that preserves date-activity relationships"""
        text = self.normalize_dates(text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_date_context = ""
        
        for sentence in sentences:
            has_date = any(re.search(pattern, sentence) for pattern in self.date_patterns)
            
            if has_date:
                current_date_context = sentence
            
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                final_chunk = current_chunk
                if current_date_context and current_date_context not in current_chunk:
                    final_chunk = f"Date context: {current_date_context}\n{current_chunk}"
                
                chunks.append(final_chunk.strip())
                
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = f"{current_date_context} {overlap_text} {sentence}"
            else:
                current_chunk += " " + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def process_document(self, document: Dict) -> List[Dict]:
        """Process document into chunks with embeddings - MISSING METHOD"""
        chunks = self.chunk_text_smart(document['content'])
        embeddings = self.generate_embeddings(chunks)
        
        processed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            processed_chunks.append({
                'chunk_id': f"{document['filename']}_{i}",
                'filename': document['filename'],
                'content': chunk,
                'embedding': embedding,
                'chunk_index': i
            })
        
        return processed_chunks
