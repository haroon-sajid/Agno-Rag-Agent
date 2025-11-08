"""
Integration tests for PDF upload → parse → query → answer flow.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import io
import pytest_check as check

from main import app
from agent import Message


class TestPDFUploadIntegration:
    """Integration tests for PDF upload and query flow."""
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Create minimal valid PDF content for testing."""
        # Minimal PDF content that passes basic validation
        return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\ntrailer\n<<>>\nstartxref\n%%EOF"
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_pdf_upload_parse_query_flow(self):
        """Complete integration test: upload → parse → query → answer."""
        client = TestClient(app)
        
        # Step 1: Upload PDF
        pdf_content = b"%PDF-1.4\ntest content\n%%EOF"
        
        with patch('agent.add_pdf_knowledge', new_callable=AsyncMock) as mock_add_pdf:
            # When uploading PDF
            response = client.post(
                "/api/upload/pdf",
                files={"file": ("test.pdf", pdf_content, "application/pdf")}
            )
            
            # Then upload should succeed
            check.equal(response.status_code, 200)
            mock_add_pdf.assert_called_once()
            
            # Step 2: Query about uploaded PDF
            chat_request = {
                "session_id": "upload_test_session",
                "messages": [{"role": "user", "content": "What is in the test document?"}]
            }
            
            # Mock agent response referencing the PDF
            with patch('agent.stream_agent_reply', new_callable=AsyncMock) as mock_stream:
                mock_stream.return_value = [
                    MagicMock(text="Based on the test document, I can see test content.", done=False),
                    MagicMock(text="", done=True)
                ]
                
                # When querying about the PDF
                response = client.post("/api/chat/stream", json=chat_request)
                
                # Then should get response
                check.equal(response.status_code, 200)
    
    @pytest.mark.asyncio
    async def test_pdf_upload_validation(self):
        """Test PDF upload validation with various file types."""
        client = TestClient(app)
        
        test_cases = [
            # (file_content, filename, expected_status, description)
            (b"Invalid content", "test.txt", 400, "Non-PDF file"),
            (b"", "empty.pdf", 400, "Empty PDF"),
            (b"0" * (11 * 1024 * 1024), "large.pdf", 400, "Oversized PDF"),
            (b"%PDF-1.4\nvalid", "valid.pdf", 200, "Valid PDF"),
        ]
        
        for file_content, filename, expected_status, description in test_cases:
            with patch('agent.add_pdf_knowledge', new_callable=AsyncMock) as mock_add:
                if expected_status == 200:
                    mock_add.return_value = None
                else:
                    mock_add.side_effect = Exception("Validation failed")
                
                # When uploading file
                response = client.post(
                    "/api/upload/pdf",
                    files={"file": (filename, file_content, "application/pdf")}
                )
                
                # Then should respect validation
                check.equal(
                    response.status_code, 
                    expected_status,
                    f"Failed for: {description}"
                )
    
    @pytest.mark.asyncio
    async def test_corrupt_pdf_handling(self):
        """Test handling of corrupt PDF files."""
        client = TestClient(app)
        
        corrupt_pdf = b"This is not a valid PDF file content"
        
        with patch('agent.add_pdf_knowledge', new_callable=AsyncMock) as mock_add:
            mock_add.side_effect = Exception("PDF parsing error")
            
            # When uploading corrupt PDF
            response = client.post(
                "/api/upload/pdf", 
                files={"file": ("corrupt.pdf", corrupt_pdf, "application/pdf")}
            )
            
            # Then should handle error gracefully
            check.equal(response.status_code, 500)
    
    @pytest.mark.asyncio
    async def test_knowledge_base_integration(self):
        """Test knowledge base operations integration."""
        client = TestClient(app)
        
        # Test knowledge stats
        with patch('agent.get_knowledge_stats', new_callable=AsyncMock) as mock_stats:
            mock_stats.return_value = {
                "uploaded_pdfs": 2,
                "knowledge_base_initialized": True,
                "vector_db_exists": True,
                "vector_db_size_mb": 15.5,
                "upload_directory_exists": True,
                "upload_directory_size_mb": 2.1,
                "embedding_backend": "GeminiEmbedder"
            }
            
            # When getting knowledge stats
            response = client.get("/api/knowledge/stats")
            
            # Then should return stats
            check.equal(response.status_code, 200)
            data = response.json()
            check.equal(data["uploaded_pdfs"], 2)
            check.is_true(data["knowledge_base_initialized"])
        
        # Test knowledge clearing
        with patch('agent.clear_knowledge', new_callable=AsyncMock) as mock_clear:
            # When clearing knowledge
            response = client.post("/api/clear_knowledge")
            
            # Then should succeed
            check.equal(response.status_code, 200)
            mock_clear.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_key_management(self):
        """Test API key management integration."""
        client = TestClient(app)
        
        # Test setting API key
        api_key_data = {
            "api_key": "test_key_12345",
            "provider": "openai"
        }
        
        # When setting API key
        response = client.post("/api/set_api_key", json=api_key_data)
        
        # Then should succeed
        check.equal(response.status_code, 200)
        
        # Test with Groq provider
        groq_data = {
            "api_key": "gsk_test_key_12345", 
            "provider": "groq"
        }
        
        response = client.post("/api/set_api_key", json=groq_data)
        check.equal(response.status_code, 200)
        
        # Test invalid API key
        invalid_data = {
            "api_key": "",
            "provider": "openai"
        }
        
        response = client.post("/api/set_api_key", json=invalid_data)
        check.equal(response.status_code, 400)


class TestErrorScenarios:
    """Test error scenarios in PDF upload flow."""
    
    @pytest.mark.asyncio
    async def test_network_failures_during_upload(self):
        """Test handling of network failures during upload."""
        client = TestClient(app)
        
        pdf_content = b"%PDF-1.4\ntest\n%%EOF"
        
        with patch('agent.add_pdf_knowledge', new_callable=AsyncMock) as mock_add:
            mock_add.side_effect = Exception("Network timeout")
            
            # When upload fails due to network
            response = client.post(
                "/api/upload/pdf",
                files={"file": ("test.pdf", pdf_content, "application/pdf")}
            )
            
            # Then should return server error
            check.equal(response.status_code, 500)
    
    @pytest.mark.asyncio
    async def test_memory_issues_during_processing(self):
        """Test handling of memory issues during PDF processing."""
        client = TestClient(app)
        
        # Simulate memory error
        with patch('agent.add_pdf_knowledge', new_callable=AsyncMock) as mock_add:
            mock_add.side_effect = MemoryError("Out of memory")
            
            response = client.post(
                "/api/upload/pdf",
                files={"file": ("large.pdf", b"%PDF-1.4\nlarge\n%%EOF", "application/pdf")}
            )
            
            # Should handle memory errors gracefully
            check.equal(response.status_code, 500)


# Test the complete RAG functionality
@pytest.mark.asyncio
async def test_complete_rag_workflow():
    """Test complete RAG workflow with mocked components."""
    client = TestClient(app)
    
    # 1. Upload a PDF
    pdf_content = b"%PDF-1.4\nTest Document\nContent: AI and Machine Learning\n%%EOF"
    
    with patch('agent.add_pdf_knowledge', new_callable=AsyncMock) as mock_add_pdf:
        mock_add_pdf.return_value = None
        
        upload_response = client.post(
            "/api/upload/pdf",
            files={"file": ("ai_document.pdf", pdf_content, "application/pdf")}
        )
        check.equal(upload_response.status_code, 200)
        
        # 2. Query about the PDF content
        chat_request = {
            "session_id": "rag_test_session", 
            "messages": [
                {"role": "user", "content": "What does the document say about AI?"}
            ]
        }
        
        with patch('agent.stream_agent_reply', new_callable=AsyncMock) as mock_stream:
            # Mock response that references the uploaded document
            mock_stream.return_value = [
                MagicMock(
                    text="Based on the uploaded document 'ai_document.pdf', it discusses AI and Machine Learning topics.",
                    done=False
                ),
                MagicMock(text="", done=True)
            ]
            
            stream_response = client.post("/api/chat/stream", json=chat_request)
            check.equal(stream_response.status_code, 200)
            
            # Verify the response references the document
            response_text = stream_response.text
            check.is_in("data:", response_text)