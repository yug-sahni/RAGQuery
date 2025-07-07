import re
import numpy as np
from typing import List, Dict
from text_processor import TextProcessor
from vector_database import VectorDatabase
from llm_handler import OllamaLLM, HuggingFaceLLM
from hybrid_search import HybridSearchEngine


class QASystem:
    def __init__(self, use_ollama: bool = True):
        self.text_processor = TextProcessor()
        self.vector_db = VectorDatabase()
        self.hybrid_search = HybridSearchEngine()
        
        # Initialize LLM
        try:
            if use_ollama:
                self.llm = OllamaLLM()
                self.llm_type = "Ollama"
            else:
                self.llm = HuggingFaceLLM()
                self.llm_type = "HuggingFace"
        except Exception as e:
            print(f"Error initializing primary LLM: {e}")
            print("Falling back to HuggingFace...")
            self.llm = HuggingFaceLLM()
            self.llm_type = "HuggingFace"
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        try:
            embedding_dim = self.text_processor.embedding_model.get_sentence_embedding_dimension()
        except:
            embedding_dim = 384  # Default dimension
        
        return {
            'llm_type': self.llm_type,
            'embedding_dimension': embedding_dim,
            'vector_db_stats': self.vector_db.get_stats()
        }
    
    def add_documents(self, processed_chunks: List[Dict]):
        """Add documents to both vector and hybrid search"""
        # Add to vector database
        self.vector_db.add_chunks(processed_chunks)
        
        # Index for hybrid search
        for chunk in processed_chunks:
            self.hybrid_search.index_chunk(chunk)
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """Answer a question using semantic search only"""
        # Generate embedding for the question
        question_embedding = self.text_processor.generate_embeddings([question])[0]
        
        # Search for relevant chunks
        search_results = self.vector_db.search(question_embedding, k=top_k)
        
        # Prepare context from retrieved chunks
        context_chunks = []
        sources = []
        
        for chunk_data, score in search_results:
            context_chunks.append(chunk_data['content'])
            sources.append({
                'filename': chunk_data['filename'],
                'chunk_index': chunk_data['chunk_index'],
                'relevance_score': score
            })
        
        context = "\n\n".join(context_chunks)
        
        # Generate answer using LLM
        prompt = f"""Based on the following context from documents, please answer the question clearly and concisely.
If the answer cannot be found in the context, please say "I cannot find this information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate_response(prompt, max_tokens=400)
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context_used': context_chunks,
            'llm_used': self.llm_type,
            'search_method': 'semantic'
        }
    
    def answer_question_enhanced(self, question: str, top_k: int = 3) -> Dict:
        """Enhanced Q&A with hybrid search for date queries"""
        question_lower = question.lower()
        
        # Check if this is a date-based query
        date_indicators = [
            'what was done on', 'activities on', 'happened on', 'occurred on',
            'on ', 'date', 'what took place on', 'events on', 'work on'
        ]
        is_date_query = any(indicator in question_lower for indicator in date_indicators)
        
        # Also check for specific date patterns in the question
        date_patterns = [
            r'\b\d{1,2}-[A-Za-z]{3,9}\b',  # 6-Sept
            r'\b[A-Za-z]{3,9}\s+\d{1,2}\b',  # Sept 6
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # 6/9/2024
        ]
        
        has_date_pattern = any(re.search(pattern, question, re.IGNORECASE) for pattern in date_patterns)
        is_date_query = is_date_query or has_date_pattern
        
        if is_date_query:
            print(f"DEBUG: Detected date query: {question}")
            
            # Use hybrid search for date queries
            date_chunk_ids = self.hybrid_search.search_by_date(question)
            print(f"DEBUG: Found chunk IDs: {date_chunk_ids}")
            
            if date_chunk_ids:
                # Get chunks by ID from hybrid search
                relevant_chunks = []
                for chunk_id in date_chunk_ids:
                    if chunk_id in self.hybrid_search.chunks_by_id:
                        chunk_data = self.hybrid_search.chunks_by_id[chunk_id]
                        relevant_chunks.append((chunk_data, 1.0))  # High confidence score
                        print(f"DEBUG: Found chunk: {chunk_data['content'][:100]}...")
                
                search_results = relevant_chunks[:top_k]
                search_method = 'hybrid'
            else:
                print("DEBUG: No date chunks found, falling back to semantic search")
                # Fallback to semantic search
                question_embedding = self.text_processor.generate_embeddings([question])[0]
                search_results = self.vector_db.search(question_embedding, k=top_k)
                search_method = 'semantic_fallback'
        else:
            # Use semantic search for technical queries
            question_embedding = self.text_processor.generate_embeddings([question])[0]
            search_results = self.vector_db.search(question_embedding, k=top_k)
            search_method = 'semantic'
        
        # Prepare context from retrieved chunks
        context_chunks = []
        sources = []
        
        for chunk_data, score in search_results:
            context_chunks.append(chunk_data['content'])
            sources.append({
                'filename': chunk_data['filename'],
                'chunk_index': chunk_data['chunk_index'],
                'relevance_score': score
            })
        
        context = "\n\n".join(context_chunks)
        
        # Enhanced prompt for date queries
        if is_date_query:
            prompt = f"""Based on the following context, answer the question about what happened on a specific date.
Pay special attention to dates and their associated activities. Look for exact date matches in the format like "6-Sept", "Sept 6", etc.

Context:
{context}

Question: {question}

Instructions:
- Look carefully for the specific date mentioned in the question
- If you find activities on that exact date, describe them in detail
- Include specific technical details, measurements, and procedures
- If the date is not found, clearly state that no information is available for that specific date

Answer:"""
        else:
            prompt = f"""Based on the following context from documents, please answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate_response(prompt, max_tokens=500)
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context_used': context_chunks,
            'search_method': search_method,
            'llm_used': self.llm_type
        }
    
    def answer_question_with_continuation(self, question: str, top_k: int = 3) -> Dict:
        """Answer question with automatic continuation if truncated"""
        initial_result = self.answer_question_enhanced(question, top_k)
        
        # Check if response seems truncated
        answer = initial_result['answer']
        
        # Signs of truncation: ends mid-sentence, no proper conclusion
        truncation_indicators = [
            not answer.strip().endswith(('.', '!', '?', ':')),
            len(answer.split()) > 100 and not any(word in answer.lower() for word in ['conclusion', 'summary', 'finally']),
            answer.endswith(('...', 'and', 'or', 'but', 'however'))
        ]
        
        if any(truncation_indicators):
            print("DEBUG: Response appears truncated, attempting continuation...")
            # Continue the response
            continuation_prompt = f"""Continue the following response naturally and complete it:

{answer}

Please continue and complete this response properly with a clear conclusion."""
            
            continuation = self.llm.generate_response(continuation_prompt, max_tokens=300)
            answer = answer + " " + continuation
        
        initial_result['answer'] = answer
        return initial_result
    
    def check_response_completeness(self, response: str) -> bool:
        """Check if response appears complete"""
        # Check for proper ending
        if not response.strip().endswith(('.', '!', '?', ':')):
            return False
        
        # Check for minimum length (very short responses might be truncated)
        if len(response.split()) < 10:
            return False
        
        # Check for incomplete sentences
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-2].split()) < 3:
            return False
        
        return True
    
    def search_similar_questions(self, question: str, limit: int = 3) -> List[Dict]:
        """Find similar questions from chat history"""
        # This would be implemented if you want to show related questions
        # For now, return empty list
        return []
    
    def get_document_summary(self) -> Dict:
        """Get a summary of all processed documents"""
        stats = self.vector_db.get_stats()
        
        # Get unique filenames
        filenames = set()
        for chunk in self.vector_db.chunks:
            filenames.add(chunk['filename'])
        
        return {
            'total_documents': len(filenames),
            'total_chunks': stats['total_chunks'],
            'document_names': list(filenames),
            'average_chunks_per_document': stats['total_chunks'] / len(filenames) if filenames else 0
        }
    
    def search_by_filename(self, filename: str, question: str, top_k: int = 3) -> Dict:
        """Search within a specific document"""
        # Filter chunks by filename
        filename_chunks = [chunk for chunk in self.vector_db.chunks if chunk['filename'] == filename]
        
        if not filename_chunks:
            return {
                'question': question,
                'answer': f"No document found with filename: {filename}",
                'sources': [],
                'context_used': [],
                'search_method': 'filename_filter',
                'llm_used': self.llm_type
            }
        
        # Generate embedding for the question
        question_embedding = self.text_processor.generate_embeddings([question])[0]
        
        # Calculate similarities for filename-filtered chunks
        similarities = []
        for chunk in filename_chunks:
            # Calculate cosine similarity
            import numpy as np
            similarity = np.dot(question_embedding, chunk['embedding']) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(chunk['embedding'])
            )
            similarities.append((chunk, float(similarity)))
        
        # Sort by similarity and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        search_results = similarities[:top_k]
        
        # Prepare context
        context_chunks = []
        sources = []
        
        for chunk_data, score in search_results:
            context_chunks.append(chunk_data['content'])
            sources.append({
                'filename': chunk_data['filename'],
                'chunk_index': chunk_data['chunk_index'],
                'relevance_score': score
            })
        
        context = "\n\n".join(context_chunks)
        
        prompt = f"""Based on the following context from the document "{filename}", please answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate_response(prompt, max_tokens=400)
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context_used': context_chunks,
            'search_method': 'filename_filter',
            'llm_used': self.llm_type,
            'filtered_by': filename
        }
