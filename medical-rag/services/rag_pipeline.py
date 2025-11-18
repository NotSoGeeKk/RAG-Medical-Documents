from typing import List, Dict, Tuple, Optional
import os
from services.vector_store import VectorStore


# Optional: Import reranking model (bonus feature)
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

# LLM client - using OpenAI as default but easily swappable
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class RAGPipeline:
    """Enhanced RAG pipeline with advanced reranking strategies"""
    
    def __init__(
        self, 
        vector_store: VectorStore,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: Vector store instance
            reranker_model: Name of cross-encoder model for reranking
        """
        self.vector_store = vector_store
        
        # Initialize reranker if available (using better model)
        self.reranker = None
        if RERANKER_AVAILABLE:
            try:
                print(f"Loading reranker model: {reranker_model}...")
                # Using L-12 instead of L-6 for better quality
                # Options: 
                # - 'cross-encoder/ms-marco-MiniLM-L-12-v2' (better quality)
                # - 'cross-encoder/ms-marco-TinyBERT-L-2-v2' (faster)
                # - 'BAAI/bge-reranker-base' (state-of-the-art)
                self.reranker = CrossEncoder(reranker_model)
            except Exception as e:
                print(f"Could not load reranker: {e}")
                print("Falling back to no reranking")
        
        # Initialize LLM client
        self.llm_client = None
        if LLM_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = OpenAI(api_key=api_key)
            else:
                print("Warning: OPENAI_API_KEY not found. Will use template-based responses.")
        
    def query(
        self, 
        question: str, 
        top_k: int = 5, 
        use_reranking: bool = True,
        use_rrf: bool = False,
        use_contextual_compression: bool = False,
        hybrid_alpha: float = 0.5
    ) -> Dict:
        """
        Answer question using RAG with advanced retrieval strategies.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            use_reranking: Whether to use cross-encoder reranking
            use_rrf: Whether to use Reciprocal Rank Fusion (combines multiple retrieval methods)
            use_contextual_compression: Whether to extract only relevant sentences
            hybrid_alpha: Weight for semantic vs keyword search (0-1)
            
        Returns:
            Dictionary with answer and sources
        """
        # Retrieve relevant chunks
        if use_rrf:
            # RRF: Combine results from multiple retrieval strategies
            results = self._reciprocal_rank_fusion(question, top_k)
        else:
            # Standard retrieval
            results = self.vector_store.search(
                question, 
                top_k=top_k * 3 if use_reranking else top_k,
                hybrid_alpha=hybrid_alpha,
                use_mmr=False  # Can enable for diversity
            )
        
        if not results:
            return {
                "answer": "No relevant information found. Please ingest documents first.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Reranking with cross-encoder
        if use_reranking and self.reranker:
            results = self._rerank(question, results, top_k * 2)
        
        # Contextual compression (extract relevant sentences)
        if use_contextual_compression and self.reranker:
            results = self._contextual_compression(question, results, top_k)
        else:
            results = results[:top_k]
        
        # Generate answer
        answer = self._generate_answer(question, results)
        
        # Prepare sources
        sources = []
        for chunk, score in results:
            sources.append({
                "text": chunk['text'][:200] + "...",  # Truncate for response
                "page": chunk['metadata'].get('page'),
                "score": float(score),
                "doc_id": chunk.get('doc_id', 'unknown')
            })
        
        # Calculate confidence (average of top scores)
        avg_confidence = sum(score for _, score in results) / len(results) if results else 0.0
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": float(avg_confidence)
        }
    
    def _rerank(self, query: str, results: List, top_k: int) -> List:
        """
        Rerank results using cross-encoder with batch processing.
        
        Args:
            query: Query text
            results: Initial retrieval results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        if not self.reranker or not results:
            return results
        
        # Prepare pairs for reranking
        pairs = [[query, chunk['text']] for chunk, _ in results]
        
        # Get reranking scores (batch processing for efficiency)
        rerank_scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        
        # Combine with original results and sort
        reranked = [(results[i][0], float(score)) for i, score in enumerate(rerank_scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def _reciprocal_rank_fusion(
        self, 
        query: str, 
        top_k: int, 
        k: int = 60
    ) -> List[Tuple[Dict, float]]:
        """
        Reciprocal Rank Fusion (RRF) combines multiple retrieval methods.
        RRF formula: score(d) = sum(1 / (k + rank_i(d))) for all ranking methods i
        
        Args:
            query: Query text
            top_k: Number of results to return
            k: RRF constant (typically 60)
            
        Returns:
            Fused results
        """
        # Get results from multiple retrieval strategies
        
        # 1. Pure semantic search
        semantic_results = self.vector_store.search(
            query, 
            top_k=top_k * 3,
            hybrid_alpha=1.0,  # Pure semantic
            use_mmr=False
        )
        
        # 2. Pure keyword search (BM25)
        keyword_results = self.vector_store.search(
            query,
            top_k=top_k * 3,
            hybrid_alpha=0.0,  # Pure BM25
            use_mmr=False
        )
        
        # 3. Hybrid search
        hybrid_results = self.vector_store.search(
            query,
            top_k=top_k * 3,
            hybrid_alpha=0.5,  # Balanced
            use_mmr=False
        )
        
        # 4. MMR for diversity
        mmr_results = self.vector_store.search(
            query,
            top_k=top_k * 3,
            hybrid_alpha=0.5,
            use_mmr=True
        )
        
        # Compute RRF scores
        rrf_scores = {}
        
        for results_list in [semantic_results, keyword_results, hybrid_results, mmr_results]:
            for rank, (chunk, _) in enumerate(results_list, start=1):
                chunk_id = chunk['chunk_id']
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = {'chunk': chunk, 'score': 0.0}
                rrf_scores[chunk_id]['score'] += 1.0 / (k + rank)
        
        # Sort by RRF score
        fused_results = [(data['chunk'], data['score']) for data in rrf_scores.values()]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results[:top_k * 2]
    
    def _contextual_compression(
        self, 
        query: str, 
        results: List[Tuple[Dict, float]], 
        top_k: int
    ) -> List[Tuple[Dict, float]]:
        """
        Contextual compression: extract only relevant sentences from chunks.
        This reduces context size and improves relevance.
        
        Args:
            query: Query text
            results: Retrieved chunks
            top_k: Number of results to return
            
        Returns:
            Compressed results with only relevant sentences
        """
        if not self.reranker:
            return results[:top_k]
        
        compressed_results = []
        
        for chunk, chunk_score in results[:top_k * 2]:
            # Split chunk into sentences
            sentences = self._split_sentences(chunk['text'])
            
            if len(sentences) <= 2:
                # Too short to compress
                compressed_results.append((chunk, chunk_score))
                continue
            
            # Score each sentence
            sentence_pairs = [[query, sent] for sent in sentences]
            sentence_scores = self.reranker.predict(sentence_pairs, batch_size=32, show_progress_bar=False)
            
            # Keep top sentences (at least 2, up to half of total)
            scored_sentences = list(zip(sentences, sentence_scores))
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            keep_count = max(2, len(sentences) // 2)
            top_sentences = scored_sentences[:keep_count]
            
            # Reconstruct chunk with only relevant sentences
            # Keep original order
            relevant_sents = set(sent for sent, _ in top_sentences)
            compressed_text = ' '.join([s for s in sentences if s in relevant_sents])
            
            # Create new chunk with compressed text
            compressed_chunk = chunk.copy()
            compressed_chunk['text'] = compressed_text
            compressed_chunk['metadata'] = chunk['metadata'].copy()
            compressed_chunk['metadata']['compressed'] = True
            compressed_chunk['metadata']['original_size'] = len(chunk['text'])
            compressed_chunk['metadata']['compressed_size'] = len(compressed_text)
            
            compressed_results.append((compressed_chunk, chunk_score))
        
        return compressed_results[:top_k]
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitter.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_answer(self, question: str, context_chunks: List) -> str:
        """
        Generate answer using LLM or template.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated answer
        """
        # Prepare context
        context = "\n\n".join([
            f"[Page {chunk['metadata'].get('page', 'N/A')}]: {chunk['text']}"
            for chunk, _ in context_chunks
        ])
        
        # If LLM is available, use it
        if self.llm_client:
            return self._generate_with_llm(question, context)
        else:
            return self._generate_template_answer(question, context_chunks)
    
    def _generate_with_llm(self, question: str, context: str) -> str:
        """
        Generate answer using OpenAI API.
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        try:
            prompt = f"""You are a medical information assistant. Answer the question based ONLY on the provided context from medical documents.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context
- If the context doesn't contain enough information, say so
- Include relevant page numbers when citing information
- Be concise but thorough
- Use medical terminology appropriately

Answer:"""
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful medical information assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer with LLM: {e}")
            return self._generate_template_answer(question, [({"text": context, "metadata": {}}, 1.0)])
    
    def _generate_template_answer(self, question: str, context_chunks: List) -> str:
        """
        Generate template-based answer when LLM is not available.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            
        Returns:
            Template answer
        """
        if not context_chunks:
            return "No relevant information found in the documents."
        
        # Extract relevant snippets
        top_chunk = context_chunks[0][0]
        page = top_chunk['metadata'].get('page', 'N/A')
        
        answer = f"""Based on the medical documents (Page {page}):

{top_chunk['text'][:500]}...

Found {len(context_chunks)} relevant sections across the document(s). 

Note: For more accurate answers, configure an LLM API key (OpenAI, Gemini, etc.) in your environment."""
        
        return answer