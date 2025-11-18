"""
Unit tests for FastAPI endpoints.

Run with: pytest tests/
"""
import pytest  
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)


def test_root_endpoint():
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_stats_endpoint():
    """Test statistics endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "embedding_dimension" in data


def test_query_without_documents():
    """Test query endpoint when no documents are ingested"""
    response = client.post(
        "/query",
        json={"question": "What is hypertension?"}
    )
    # Should return 400 if no documents
    assert response.status_code in [200, 400]


def test_query_empty_question():
    """Test query with empty question"""
    response = client.post(
        "/query",
        json={"question": ""}
    )
    assert response.status_code == 400


def test_ingest_invalid_file():
    """Test ingest with non-PDF file"""
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", b"test content", "text/plain")}
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


# Note: Full integration tests would require a sample PDF file
# Add more tests as needed for your specific use cases

if __name__ == "__main__":
    pytest.main([__file__, "-v"])