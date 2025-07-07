import faiss
import pickle
from typing import List, Tuple, Dict
import numpy as np
import os


class VectorDatabase:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunks = []
        
    def add_chunks(self, processed_chunks: List[Dict]):
        """Add document chunks to the vector database"""
        embeddings = np.array([chunk['embedding'] for chunk in processed_chunks])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.chunks.extend(processed_chunks)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar chunks"""
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, filepath: str):
        """Save the vector database"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': faiss.serialize_index(self.index),
                'chunks': self.chunks,
                'dimension': self.dimension
            }, f)
    
    def load(self, filepath: str):
        """Load the vector database"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = faiss.deserialize_index(data['index'])
            self.chunks = data['chunks']
            self.dimension = data['dimension']
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_chunks': len(self.chunks),
            'unique_documents': len(set(chunk['filename'] for chunk in self.chunks)),
            'dimension': self.dimension
        }
    
