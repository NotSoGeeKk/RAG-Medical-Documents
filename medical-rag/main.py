from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from pathlib import Path

from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStore
from services.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Medical PDF RAG System",
    description="RAG system for medical document Q&A",
    version="1.0.0"
)

# Initialize services with improved strategies
pdf_processor = PDFProcessor(
    chunk_size=512,
    chunk_overlap=128,
    chunking_strategy="semantic"  # Options: "simple", "semantic", "hierarchical"
)
vector_store = VectorStore(
    model_name="all-MiniLM-L6-v2",
    use_hybrid_search=True  # Enable hybrid search (semantic + BM25)
)
rag_pipeline = RAGPipeline(
    vector_store,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2"  # Better reranker
)

# Storage directory for uploaded PDFs
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_reranking: Optional[bool] = True  # Enable by default
    use_rrf: Optional[bool] = False  # Reciprocal Rank Fusion
    use_contextual_compression: Optional[bool] = False  # Extract relevant sentences
    hybrid_alpha: Optional[float] = 0.5  # Weight for semantic vs keyword (0=BM25, 1=semantic)


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: Optional[float] = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Medical PDF RAG System",
        "endpoints": {
            "ingest": "POST /ingest - Upload and process PDF",
            "query": "POST /query - Ask questions about documents"
        }
    }


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a medical PDF and store its embeddings in the vector database.
    
    Args:
        file: PDF file to process
        
    Returns:
        Success message with processing statistics
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text from PDF with metadata
        print(f"Processing PDF: {file.filename}")
        chunks = pdf_processor.process_pdf(str(file_path))
        
        if not chunks:
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from the PDF"
            )
        
        # Store in vector database
        doc_id = vector_store.add_documents(chunks, file.filename)
        
        # Get statistics
        stats = {
            "document_id": doc_id,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "status": "success",
            "message": f"Successfully processed {len(chunks)} chunks from {file.filename}"
        }
        
        print(f"Ingestion complete: {stats}")
        return JSONResponse(content=stats, status_code=200)
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document database using natural language.
    
    Args:
        request: Query parameters including question and options
        
    Returns:
        Answer generated from relevant document chunks
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Check if vector store has documents
        if vector_store.is_empty():
            raise HTTPException(
                status_code=400, 
                detail="No documents in database. Please ingest documents first using /ingest endpoint"
            )
        
        # Generate answer using RAG pipeline
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            use_rrf=request.use_rrf,
            use_contextual_compression=request.use_contextual_compression,
            hybrid_alpha=request.hybrid_alpha
        )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.delete("/clear")
async def clear_database():
    """Clear all documents from the vector database"""
    try:
        vector_store.clear()
        return {"status": "success", "message": "Vector database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get statistics about the current database"""
    try:
        stats = vector_store.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )