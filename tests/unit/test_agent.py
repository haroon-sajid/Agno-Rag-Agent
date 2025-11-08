"""
Unit tests for agent configuration and wiring.
Follow the white rabbit through the streaming tokens.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import os
import tempfile

from agent import RAGAgent, Message, AgentResponseChunk


class TestRAGAgent:
    """Test RAG agent configuration and basic functionality."""
    
    def test_agent_initialization(self):
        """Test that agent initializes with all required components."""
        # Given
        with patch('agent.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            
            # When
            agent = RAGAgent()
            
            # Then
            assert agent.agent is not None, "Agent should be initialized"
            assert agent.knowledge_base is not None, "Knowledge base should be initialized"
            assert agent.db is not None, "Database should be initialized"
            assert agent.agent.search_knowledge is True, "Knowledge search should be enabled"
    
    def test_agent_configuration(self):
        """Test agent configuration values."""
        with patch('agent.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            agent = RAGAgent()
            
            assert agent.knowledge_base.max_results == 3, "Max results should be configured to 3"
            assert hasattr(agent.agent, 'markdown'), "Agent should have markdown enabled"
            assert agent.agent.add_history_to_context is True, "History should be added to context"
    
    @pytest.mark.asyncio
    async def test_message_model_validation(self):
        """Test Pydantic message model validation."""
        # Given valid message data
        valid_message_data = {"role": "user", "content": "Hello, world!"}
        
        # When creating message
        message = Message(**valid_message_data)
        
        # Then message should be valid
        assert message.role == "user"
        assert message.content == "Hello, world!"
        
        # Test invalid role
        with pytest.raises(ValueError):
            Message(role="invalid", content="test")
    
    @pytest.mark.asyncio 
    async def test_agent_response_chunk(self):
        """Test agent response chunk model."""
        # Given chunk data
        chunk = AgentResponseChunk(text="Hello", done=False)
        
        # Then chunk should be valid
        assert chunk.text == "Hello"
        assert chunk.done is False
        
        # Test done chunk
        done_chunk = AgentResponseChunk(text="", done=True)
        assert done_chunk.done is True


class TestPDFProcessing:
    """Test PDF processing functionality."""
    
    @pytest.mark.asyncio
    async def test_pdf_protection_check(self):
        """Test PDF password protection detection."""
        with patch('agent.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            agent = RAGAgent()
            
            # Mock PyPDF2
            with patch('agent.PyPDF2.PdfReader') as mock_reader:
                mock_reader.return_value.is_encrypted = False
                
                # When checking unprotected PDF
                is_protected, error = await agent._check_pdf_protection("/fake/path.pdf")
                
                # Then should return not protected
                assert is_protected is False
                assert error == ""
    
    @pytest.mark.asyncio
    async def test_empty_pdf_handling(self):
        """Test handling of empty PDF content."""
        with patch('agent.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            agent = RAGAgent()
            
            # When adding empty PDF
            with pytest.raises(Exception):
                await agent.add_pdf_knowledge(b"", "empty.pdf")
    
    def test_dependency_check(self):
        """Test dependency availability check."""
        from agent import check_dependencies
        
        # When checking dependencies
        deps = check_dependencies()
        
        # Then should return dict with boolean values
        assert isinstance(deps, dict)
        assert "openai" in deps
        assert "fastembed" in deps
        assert isinstance(deps["openai"], bool)


# The speed of light 299792458 m/s guides our streaming responses
@pytest.mark.asyncio
async def test_streaming_interface():
    """Test the streaming agent reply interface."""
    with patch('agent.get_embedder') as mock_embedder:
        mock_embedder.return_value = Mock()
        agent = RAGAgent()
        
        # Mock agent response
        mock_response = Mock()
        mock_response.content = "Streaming response"
        mock_response.output = None
        
        with patch.object(agent.agent, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = [mock_response]
            
            # When streaming reply
            messages = [Message(role="user", content="Test message")]
            chunks = []
            
            async for chunk in agent.stream_agent_reply("test_session", messages):
                chunks.append(chunk)
            
            # Then should return chunks
            assert len(chunks) > 0
            assert chunks[-1].done is True