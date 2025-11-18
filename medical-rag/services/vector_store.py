import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
import re


class VectorStore:
    """Hybrid vector store with FAISS (semantic) and BM25 (keyword) search"""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        cache_dir: str = "cache",
        use_hybrid_search: bool = True
    ):
        """
        Initialize hybrid vector store with embedding model and BM25.
        
        Args:
            model_name: Name of SentenceTransformer model
            cache_dir: Directory for caching index and metadata
            use_hybrid_search: Whether to enable hybrid search (semantic + BM25)
        """
        print(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.use_hybrid_search = use_hybrid_search
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Initialize BM25 for keyword search
        self.bm25 = None
        self.tokenized_corpus = []
        
        # Store document metadata
        self.documents = []
        self.document_map = {}  # Maps doc_id to list of chunk indices
        
        # Caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index_path = self.cache_dir / "faiss_index.bin"
        self.metadata_path = self.cache_dir / "metadata.pkl"
        self.bm25_path = self.cache_dir / "bm25.pkl"
        
        # Load cached data if exists
        self._load_cache()
    
    def add_documents(self, chunks: List[Dict], doc_id: str) -> str:
        """
        Add document chunks to vector store and BM25 index.
        
        Args:
            chunks: List of text chunks with metadata
            doc_id: Document identifier
            
        Returns:
            Document ID
        """
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add to FAISS index
        start_idx = len(self.documents)
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        chunk_indices = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'doc_id': doc_id,
                'chunk_id': start_idx + i
            }
            self.documents.append(chunk_data)
            chunk_indices.append(start_idx + i)
        
        # Map document to chunks
        self.document_map[doc_id] = chunk_indices
        
        # Build/rebuild BM25 index for hybrid search
        if self.use_hybrid_search:
            self._build_bm25_index()
        
        # Save to cache
        self._save_cache()
        
        print(f"Added {len(chunks)} chunks to vector store")
        return doc_id
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        hybrid_alpha: float = 0.5,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5
    ) -> List[Tuple[Dict, float]]:
        """
        Search for relevant chunks using hybrid search (semantic + BM25).
        
        Args:
            query: Query text
            top_k: Number of results to return
            hybrid_alpha: Weight for semantic vs BM25 (0=pure BM25, 1=pure semantic)
            use_mmr: Whether to use Maximal Marginal Relevance for diversity
            mmr_lambda: MMR lambda parameter (1=relevance, 0=diversity)
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.is_empty():
            return []
        
        if self.use_hybrid_search and self.bm25 is not None:
            results = self._hybrid_search(query, top_k, hybrid_alpha)
        else:
            results = self._semantic_search(query, top_k)
        
        # Apply MMR for diversity if requested
        if use_mmr and len(results) > 0:
            results = self._maximal_marginal_relevance(query, results, top_k, mmr_lambda)
        
        return results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """
        Semantic search using FAISS embeddings.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(top_k * 2, len(self.documents))  # Get more for hybrid
        )
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                chunk = self.documents[idx]
                # Convert L2 distance to similarity score
                similarity = 1 / (1 + dist)
                results.append((chunk, similarity))
        
        return results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """
        BM25 keyword search.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.bm25 is None:
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # Prepare results with normalized scores
        max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                chunk = self.documents[idx]
                normalized_score = bm25_scores[idx] / max_score
                results.append((chunk, float(normalized_score)))
        
        return results
    
    def _hybrid_search(self, query: str, top_k: int, alpha: float = 0.5) -> List[Tuple[Dict, float]]:
        """
        Hybrid search combining semantic and BM25 search.
        
        Args:
            query: Query text
            top_k: Number of results
            alpha: Weight for semantic (1-alpha for BM25)
            
        Returns:
            List of (chunk, score) tuples sorted by combined score
        """
        # Get results from both methods
        semantic_results = self._semantic_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)
        
        # Create score dictionaries
        semantic_scores = {chunk['chunk_id']: score for chunk, score in semantic_results}
        bm25_scores = {chunk['chunk_id']: score for chunk, score in bm25_results}
        
        # Combine scores
        all_chunk_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())
        combined_results = []
        
        for chunk_id in all_chunk_ids:
            sem_score = semantic_scores.get(chunk_id, 0.0)
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            combined_score = alpha * sem_score + (1 - alpha) * bm25_score
            
            # Get the chunk
            chunk = self.documents[chunk_id]
            combined_results.append((chunk, combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    def _maximal_marginal_relevance(
        self, 
        query: str, 
        results: List[Tuple[Dict, float]], 
        top_k: int,
        lambda_param: float = 0.5
    ) -> List[Tuple[Dict, float]]:
        """
        Apply Maximal Marginal Relevance for diverse results.
        
        Args:
            query: Query text
            results: Initial search results
            top_k: Number of final results
            lambda_param: Balance between relevance and diversity
            
        Returns:
            Diverse list of results
        """
        if len(results) <= top_k:
            return results
        
        # Get embeddings for all result texts
        texts = [chunk['text'] for chunk, _ in results]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # Calculate relevance scores (cosine similarity)
        relevance_scores = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(results)))
        
        # Select first document with highest relevance
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select documents
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = relevance_scores[idx]
                
                # Maximum similarity to already selected documents
                max_sim = max([
                    np.dot(embeddings[idx], embeddings[sel_idx]) / (
                        np.linalg.norm(embeddings[idx]) * np.linalg.norm(embeddings[sel_idx])
                    )
                    for sel_idx in selected_indices
                ])
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr))
            
            # Select document with highest MMR
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return selected results
        return [results[i] for i in selected_indices]
    
    def _build_bm25_index(self):
        """Build BM25 index from current documents."""
        if not self.documents:
            return
        
        print("Building BM25 index for keyword search...")
        # Tokenize all documents
        self.tokenized_corpus = [
            self._tokenize(doc['text']) for doc in self.documents
        ]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def is_empty(self) -> bool:
        """Check if vector store is empty"""
        return len(self.documents) == 0
    
    def clear(self):
        """Clear all documents and reset indices"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.document_map = {}
        self.bm25 = None
        self.tokenized_corpus = []
        
        # Clear cache
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        if self.bm25_path.exists():
            self.bm25_path.unlink()
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            "total_chunks": len(self.documents),
            "total_documents": len(self.document_map),
            "embedding_dimension": self.dimension,
            "model": self.model_name
        }
    
    def _save_cache(self):
        """Save index, BM25, and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'document_map': self.document_map
                }, f)
            
            # Save BM25 index
            if self.bm25 is not None and self.use_hybrid_search:
                with open(self.bm25_path, 'wb') as f:
                    pickle.dump({
                        'bm25': self.bm25,
                        'tokenized_corpus': self.tokenized_corpus
                    }, f)
            
            print(f"Cache saved to {self.cache_dir}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _load_cache(self):
        """Load index, BM25, and metadata from disk"""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.document_map = data['document_map']
                
                # Load BM25 index if available
                if self.bm25_path.exists() and self.use_hybrid_search:
                    with open(self.bm25_path, 'rb') as f:
                        bm25_data = pickle.load(f)
                        self.bm25 = bm25_data['bm25']
                        self.tokenized_corpus = bm25_data['tokenized_corpus']
                
                print(f"Loaded {len(self.documents)} chunks from cache")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")