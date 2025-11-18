# Medical Document RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system specifically designed for medical documents. This system allows you to upload PDF medical documents and ask natural language questions about their content, receiving accurate answers backed by relevant document excerpts.

## 🚀 Features

- **PDF Processing**: Extract and process text from medical PDF documents with intelligent chunking strategies
- **Advanced Vector Search**: Combines semantic search (embeddings) with keyword-based BM25 for hybrid retrieval
- **Reranking**: Uses cross-encoders to improve result relevance
- **Multiple Chunking Strategies**: Semantic, hierarchical, and simple chunking options
- **FastAPI Backend**: RESTful API for document ingestion and querying
- **Contextual Compression**: Extract only the most relevant sentences from retrieved chunks
- **Comprehensive Testing**: Full test suite with pytest

## 🏗️ Architecture

The system consists of three main services:

- **PDFProcessor**: Handles PDF text extraction and intelligent document chunking
- **VectorStore**: Manages FAISS vector database with hybrid search capabilities
- **RAGPipeline**: Orchestrates the retrieval and generation process

## 📋 Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## 🛠️ Installation

1. **Clone the repository** (if applicable) and navigate to the project directory

2. **Create and activate virtual environment**:
   ```bash
   cd medical-rag
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Starting the Server

Run the FastAPI server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```http
GET /
```
Returns service status and available endpoints.

#### Upload Document
```http
POST /ingest
Content-Type: multipart/form-data
```
Upload a PDF file for processing and indexing.

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/ingest" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_medical_document.pdf"
```

#### Query Documents
```http
POST /query
Content-Type: application/json
```

Ask questions about the uploaded documents.

**Request Body**:
```json
{
  "question": "What are the side effects of this medication?",
  "top_k": 5,
  "use_reranking": true,
  "use_rrf": false,
  "use_contextual_compression": false,
  "hybrid_alpha": 0.5
}
```

**Parameters**:
- `question`: Your natural language question
- `top_k`: Number of relevant chunks to retrieve (default: 5)
- `use_reranking`: Whether to rerank results for better relevance (default: true)
- `use_rrf`: Use Reciprocal Rank Fusion (default: false)
- `use_contextual_compression`: Extract only relevant sentences (default: false)
- `hybrid_alpha`: Balance between semantic (1.0) and keyword (0.0) search (default: 0.5)

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What are the common symptoms?",
       "top_k": 3
     }'
```

#### Clear Database
```http
DELETE /clear
```
Remove all documents from the vector database.

#### Get Statistics
```http
GET /stats
```
Get information about the current document collection.

## 🧪 Testing

Run the test suite: Only to check whether the APIs are woking or not.

```bash
pytest tests/
```

## ⚙️ Configuration

### Chunking Strategies

The system supports multiple chunking strategies for PDF processing:

- **Simple**: Fixed-size chunks with overlap
- **Semantic**: Content-aware chunking based on semantic similarity
- **Hierarchical**: Multi-level chunking for better context preservation

### Embedding Models

Default model: `all-MiniLM-L6-v2`
- Lightweight and fast
- Good performance for medical text
- Can be changed in the VectorStore initialization

### Reranking Models

Default reranker: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- Improves relevance of retrieved documents
- Can be disabled for faster responses

## 📁 Project Structure

```
medical-rag/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── services/
│   ├── pdf_processor.py    # PDF text extraction and chunking
│   ├── vector_store.py     # FAISS vector database management
│   └── rag_pipeline.py     # RAG query processing pipeline
├── tests/
│   └── tests.py           # Test suite (only for ensuring whether the apis are working or not)
├── uploads/               # Uploaded PDF files
├── cache/                 # Vector store cache files
└── venv/                  # Virtual environment (created during setup)
```