import pdfplumber
from typing import List, Dict, Optional 
import re
from sentence_transformers import SentenceTransformer
import numpy as np


class PDFProcessor:
    """Handles PDF text extraction and advanced chunking with metadata"""
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 128,
        chunking_strategy: str = "semantic",  # "simple", "semantic", "hierarchical"
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """
        Initialize PDF processor with advanced chunking strategies.
        
        Args:
            chunk_size: Target size for text chunks (in tokens/characters)
            chunk_overlap: Overlap between consecutive chunks
            chunking_strategy: Strategy to use ("simple", "semantic", "hierarchical")
            embedding_model: Optional pre-loaded embedding model for semantic chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        
        # Load embedding model for semantic chunking
        self.embedding_model = embedding_model
        if chunking_strategy == "semantic" and embedding_model is None:
            print("Loading embedding model for semantic chunking...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF and create chunks with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        full_text = []
        page_boundaries = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text from page
                text = page.extract_text()
                
                if text:
                    # Clean text
                    text = self._clean_text(text)
                    full_text.append(text)
                    page_boundaries.append((page_num, len(' '.join(full_text))))
        
        # Join all text for better chunking
        combined_text = ' '.join(full_text)
        
        # Create chunks based on selected strategy
        if self.chunking_strategy == "semantic":
            chunks = self._semantic_chunking(combined_text, page_boundaries)
        elif self.chunking_strategy == "hierarchical":
            chunks = self._hierarchical_chunking(combined_text, page_boundaries)
        else:  # simple
            chunks = self._simple_chunking(combined_text, page_boundaries)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove special characters but keep medical symbols
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\%\°\±\≤\≥\→]', '', text)
        
        return text.strip()
    
    def _simple_chunking(self, text: str, page_boundaries: List) -> List[Dict[str, any]]:
        """
        Simple sentence-based chunking with overlap.
        
        Args:
            text: Full text to chunk
            page_boundaries: List of (page_num, char_position) tuples
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_sentences = []
        char_position = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                page_num = self._get_page_number(char_position, page_boundaries)
                
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        'page': page_num,
                        'chunk_size': len(current_chunk),
                        'num_sentences': len(current_sentences),
                        'chunking_strategy': 'simple'
                    }
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_sentences)
                current_chunk = overlap_text + " " + sentence
                current_sentences = [sentence]
            else:
                current_chunk += " " + sentence
                current_sentences.append(sentence)
            
            char_position += len(sentence) + 1
        
        # Add final chunk
        if current_chunk.strip():
            page_num = self._get_page_number(char_position, page_boundaries)
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'page': page_num,
                    'chunk_size': len(current_chunk),
                    'num_sentences': len(current_sentences),
                    'chunking_strategy': 'simple'
                }
            })
        
        return chunks
    
    def _semantic_chunking(self, text: str, page_boundaries: List) -> List[Dict[str, any]]:
        """
        Semantic chunking based on embedding similarity.
        Groups sentences by semantic coherence.
        
        Args:
            text: Full text to chunk
            page_boundaries: List of (page_num, char_position) tuples
            
        Returns:
            List of chunk dictionaries
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        # Generate embeddings for all sentences
        embeddings = self.embedding_model.encode(sentences, convert_to_numpy=True)
        
        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(sim)
        
        # Find split points where similarity drops
        if similarities:
            threshold = np.percentile(similarities, 25)  # Split at low similarity points
            split_indices = [0]
            
            for i, sim in enumerate(similarities):
                if sim < threshold or len(' '.join(sentences[split_indices[-1]:i+1])) > self.chunk_size:
                    split_indices.append(i + 1)
            
            split_indices.append(len(sentences))
        else:
            split_indices = [0, len(sentences)]
        
        # Create chunks
        chunks = []
        char_position = 0
        
        for i in range(len(split_indices) - 1):
            start, end = split_indices[i], split_indices[i + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = ' '.join(chunk_sentences)
            
            # Add overlap from previous chunk
            if i > 0 and split_indices[i] > 0:
                overlap_start = max(0, split_indices[i] - 2)
                overlap = ' '.join(sentences[overlap_start:split_indices[i]])
                if len(overlap) <= self.chunk_overlap:
                    chunk_text = overlap + ' ' + chunk_text
            
            page_num = self._get_page_number(char_position, page_boundaries)
            
            chunks.append({
                'text': chunk_text.strip(),
                'metadata': {
                    'page': page_num,
                    'chunk_size': len(chunk_text),
                    'num_sentences': len(chunk_sentences),
                    'chunking_strategy': 'semantic'
                }
            })
            
            char_position += len(chunk_text) + 1
        
        return chunks
    
    def _hierarchical_chunking(self, text: str, page_boundaries: List) -> List[Dict[str, any]]:
        """
        Hierarchical chunking: creates parent (large) and child (small) chunks.
        Child chunks are used for retrieval, parent chunks provide context.
        
        Args:
            text: Full text to chunk
            page_boundaries: List of (page_num, char_position) tuples
            
        Returns:
            List of chunk dictionaries with parent-child relationships
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        # Create parent chunks (larger)
        parent_size = self.chunk_size * 3
        parent_chunks = []
        current_parent = ""
        parent_sentences = []
        char_position = 0
        
        for sentence in sentences:
            if len(current_parent) + len(sentence) > parent_size and current_parent:
                page_num = self._get_page_number(char_position, page_boundaries)
                parent_chunks.append({
                    'text': current_parent.strip(),
                    'sentences': parent_sentences.copy(),
                    'page': page_num,
                    'char_position': char_position
                })
                current_parent = sentence
                parent_sentences = [sentence]
            else:
                current_parent += " " + sentence
                parent_sentences.append(sentence)
            
            char_position += len(sentence) + 1
        
        # Add last parent
        if current_parent.strip():
            page_num = self._get_page_number(char_position, page_boundaries)
            parent_chunks.append({
                'text': current_parent.strip(),
                'sentences': parent_sentences.copy(),
                'page': page_num,
                'char_position': char_position
            })
        
        # Create child chunks from each parent
        all_chunks = []
        for parent_idx, parent in enumerate(parent_chunks):
            child_sentences = parent['sentences']
            current_child = ""
            child_count = 0
            
            for sentence in child_sentences:
                if len(current_child) + len(sentence) > self.chunk_size and current_child:
                    all_chunks.append({
                        'text': current_child.strip(),
                        'metadata': {
                            'page': parent['page'],
                            'chunk_size': len(current_child),
                            'chunking_strategy': 'hierarchical',
                            'parent_text': parent['text'],
                            'parent_id': parent_idx,
                            'child_id': child_count
                        }
                    })
                    current_child = sentence
                    child_count += 1
                else:
                    current_child += " " + sentence
            
            # Add last child
            if current_child.strip():
                all_chunks.append({
                    'text': current_child.strip(),
                    'metadata': {
                        'page': parent['page'],
                        'chunk_size': len(current_child),
                        'chunking_strategy': 'hierarchical',
                        'parent_text': parent['text'],
                        'parent_id': parent_idx,
                        'child_id': child_count
                    }
                })
        
        return all_chunks
    
    def _get_page_number(self, char_position: int, page_boundaries: List) -> int:
        """Get page number for a character position."""
        for page_num, boundary in page_boundaries:
            if char_position <= boundary:
                return page_num
        return page_boundaries[-1][0] if page_boundaries else 1
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Improved sentence splitting with medical abbreviation handling.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Common medical abbreviations that shouldn't trigger splits
        medical_abbrevs = r'(?:Dr|Mr|Mrs|Ms|vs|etc|i\.e|e\.g|Fig|Tab|Vol|No|pp|approx|ca|viz|Inc|Ltd|Jr|Sr|Ph\.D|M\.D|B\.A|M\.A)'
        
        # Protect abbreviations temporarily
        protected_text = re.sub(f'({medical_abbrevs})\.', r'\1<PERIOD>', text, flags=re.IGNORECASE)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected_text)
        
        # Restore periods
        sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences if s.strip()]
        
        # Merge very short sentences (likely incomplete)
        merged_sentences = []
        current = ""
        
        for sent in sentences:
            if len(sent) < 20 and current:
                current += " " + sent
            else:
                if current:
                    merged_sentences.append(current)
                current = sent
        
        if current:
            merged_sentences.append(current)
        
        return merged_sentences
    
    def _get_overlap(self, sentences: List[str]) -> str:
        """
        Get overlap text from the end of previous chunk.
        
        Args:
            sentences: List of sentences from previous chunk
            
        Returns:
            Overlap text
        """
        if not sentences:
            return ""
        
        # Take last 1-2 sentences for overlap
        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.chunk_overlap:
                overlap_text = sentence + " " + overlap_text
            else:
                break
        
        return overlap_text.strip()