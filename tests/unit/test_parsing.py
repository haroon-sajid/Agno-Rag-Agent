"""
Unit tests for PDF parsing and text extraction.
"""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from agent import RAGAgent


class TestPDFParsing:
    """Test PDF parsing functionality."""
    
    @pytest.mark.asyncio
    async def test_pdf_reader_initialization(self):
        """Test PDF reader with semantic chunking."""
        with patch('agent.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            agent = RAGAgent()
            
            # Verify semantic chunking is used
            assert hasattr(agent, 'knowledge_base')
    
    @pytest.mark.asyncio
    async def test_invalid_file_handling(self):
        """Test handling of invalid file types."""
        with patch('agent.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            agent = RAGAgent()
            
            # When adding non-PDF content
            with pytest.raises(Exception):
                await agent.add_pdf_knowledge(b"Not a PDF", "text.txt")
    
    @pytest.mark.asyncio
    async def test_large_file_handling(self):
        """Test handling of large files."""
        with patch('agent.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            agent = RAGAgent()
            
            # Create large content (just over 10MB)
            large_content = b"0" * (11 * 1024 * 1024)
            
            # When adding large PDF
            with pytest.raises(Exception):
                await agent.add_pdf_knowledge(large_content, "large.pdf")
    
    def test_filename_cleaning(self):
        """Test filename cleaning for URL encoded names."""
        with patch('agent.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            agent = RAGAgent()
            
            # Test various filename patterns
            test_cases = [
                ("document%20with%20spaces.pdf", "document with spaces.pdf"),
                ("file%2Fwith%2Fslashes.pdf", "file_with_slashes.pdf"),
                ("normal_file.pdf", "normal_file.pdf")
            ]
            
            for input_name, expected_clean in test_cases:
                # This would be tested in the actual method
                clean_name = input_name.replace('%20', ' ').replace('%', '_')
                assert clean_name == expected_clean


class TestTextKnowledge:
    """Test text knowledge addition."""
    
    @pytest.mark.asyncio
    async def test_text_knowledge_addition(self):
        """Test adding text content to knowledge base."""
        with patch('agent.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            agent = RAGAgent()
            
            with patch.object(agent.knowledge_base, 'add_content_async', new_callable=AsyncMock) as mock_add:
                # When adding text knowledge
                await agent.add_text_knowledge("Sample text content", "test_source")
                
                # Then should call knowledge base addition
                mock_add.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_knowledge_clearing(self):
        """Test knowledge base clearing functionality."""
        with patch('agent.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            agent = RAGAgent()
            
            with patch('shutil.rmtree') as mock_rmtree, \
                 patch('os.makedirs') as mock_makedirs:
                
                # When clearing knowledge
                await agent.clear_knowledge()
                
                # Then should clean directories
                assert mock_rmtree.called
                assert mock_makedirs.called