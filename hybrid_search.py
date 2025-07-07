import re
from typing import List, Dict, Tuple
from collections import defaultdict

class HybridSearchEngine:
    def __init__(self):
        self.keyword_index = defaultdict(list)
        self.date_index = defaultdict(list)
        self.chunks_by_id = {}  # Store chunks by ID for easy retrieval
        
    def index_chunk(self, chunk_data: Dict):
        """Index chunk for both semantic and keyword search"""
        chunk_id = chunk_data['chunk_id']
        content = chunk_data['content'].lower()
        
        # Store chunk for retrieval
        self.chunks_by_id[chunk_id] = chunk_data
        
        # Extract and index dates with better normalization
        dates = self.extract_dates(chunk_data['content'])  # Use original content, not lowercased
        for date in dates:
            self.date_index[date.lower()].append(chunk_id)
        
        # Extract and index technical keywords
        keywords = self.extract_keywords(content)
        for keyword in keywords:
            self.keyword_index[keyword].append(chunk_id)
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract various date formats with comprehensive variants"""
        date_patterns = [
            r'\b\d{1,2}-[A-Za-z]{3,9}\b',  # 6-Sept, 15-December
            r'\b[A-Za-z]{3,9}\s+\d{1,2}\b',  # Sept 6, December 15
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # 6/9/2024
            r'\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b',  # September 6, 2024
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        # Generate all possible variants for each found date
        all_variants = []
        for date in dates:
            variants = self.generate_date_variants(date)
            all_variants.extend(variants)
        
        return list(set(all_variants))
    
    def generate_date_variants(self, date_str: str) -> List[str]:
        """Generate comprehensive date variants"""
        variants = [date_str]
        
        # Handle different formats
        if '-' in date_str:
            # Format: 6-Sept
            parts = date_str.split('-')
            if len(parts) == 2:
                day, month = parts
                month_mappings = {
                    'jan': ['january', 'jan'],
                    'feb': ['february', 'feb'],
                    'mar': ['march', 'mar'],
                    'apr': ['april', 'apr'],
                    'may': ['may'],
                    'jun': ['june', 'jun'],
                    'jul': ['july', 'jul'],
                    'aug': ['august', 'aug'],
                    'sep': ['september', 'sept', 'sep'],
                    'oct': ['october', 'oct'],
                    'nov': ['november', 'nov'],
                    'dec': ['december', 'dec']
                }
                
                month_lower = month.lower()[:3]  # Get first 3 characters
                if month_lower in month_mappings:
                    for month_variant in month_mappings[month_lower]:
                        variants.extend([
                            f"{day}-{month_variant}",
                            f"{day} {month_variant}",
                            f"{month_variant} {day}",
                            f"{day}-{month_variant.capitalize()}",
                            f"{day} {month_variant.capitalize()}",
                            f"{month_variant.capitalize()} {day}"
                        ])
        
        elif ' ' in date_str:
            # Handle "Sept 6" or "September 6, 2024" formats
            parts = date_str.replace(',', '').split()
            if len(parts) >= 2:
                if parts[0].isalpha() and parts[1].isdigit():
                    # Month Day format
                    month, day = parts[0], parts[1]
                    variants.extend([
                        f"{day}-{month}",
                        f"{day} {month}",
                        f"{month} {day}",
                        f"{day}-{month.lower()}",
                        f"{day} {month.lower()}",
                        f"{month.lower()} {day}"
                    ])
                elif parts[0].isdigit() and parts[1].isalpha():
                    # Day Month format
                    day, month = parts[0], parts[1]
                    variants.extend([
                        f"{day}-{month}",
                        f"{day} {month}",
                        f"{month} {day}",
                        f"{day}-{month.lower()}",
                        f"{day} {month.lower()}",
                        f"{month.lower()} {day}"
                    ])
        
        return list(set(variants))
    
    def search_by_date(self, query: str) -> List[str]:
        """Enhanced date search with better matching"""
        query_lower = query.lower()
        
        # Extract potential date terms from query
        date_patterns = [
            r'\b\d{1,2}-[A-Za-z]{3,9}\b',
            r'\b[A-Za-z]{3,9}\s+\d{1,2}\b',
            r'\b\d{1,2}\s+[A-Za-z]{3,9}\b',
            r'\bon\s+(\d{1,2}-[A-Za-z]{3,9})\b',
            r'\bon\s+([A-Za-z]{3,9}\s+\d{1,2})\b'
        ]
        
        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if isinstance(matches[0], tuple) if matches else False:
                found_dates.extend([match[0] for match in matches])
            else:
                found_dates.extend(matches)
        
        # If no explicit dates found, look for date indicators
        if not found_dates:
            # Look for month names and numbers
            month_day_pattern = r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{1,2})\b'
            day_month_pattern = r'\b(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b'
            
            month_day_matches = re.findall(month_day_pattern, query_lower)
            day_month_matches = re.findall(day_month_pattern, query_lower)
            
            for month, day in month_day_matches:
                found_dates.append(f"{month} {day}")
            
            for day, month in day_month_matches:
                found_dates.append(f"{day} {month}")
        
        # Generate variants for found dates and search
        matching_chunks = set()
        for date_term in found_dates:
            variants = self.generate_date_variants(date_term)
            for variant in variants:
                variant_lower = variant.lower()
                if variant_lower in self.date_index:
                    matching_chunks.update(self.date_index[variant_lower])
        
        return list(matching_chunks)
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract technical keywords"""
        technical_terms = [
            'drilling', 'hole', 'section', 'wbm', 'bbls', 'sweep', 'circulate',
            'weight', 'trip', 'shoe', 'rams', 'bottom', 'pill', 'gyro', 'pull'
        ]
        
        keywords = []
        for term in technical_terms:
            if term in text:
                keywords.append(term)
        
        return keywords
