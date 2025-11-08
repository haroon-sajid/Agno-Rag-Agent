"""
Integration tests for streaming functionality.
Verify multiple streamed chunks, not single blob.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
import aiohttp
import pytest_check as check

from main import app
from agent import Message, AgentResponseChunk


class TestStreamingIntegration:
    """Integration tests for streaming endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_chat_stream_multiple_chunks(self):
        """Test that streaming returns multiple chunks, not a single blob."""
        # Given a chat request
        chat_request = {
            "session_id": "test_session",
            "messages": [
                {"role": "user", "content": "Tell me a story about AI."}
            ]
        }
        
        # Mock the agent response to return multiple chunks
        mock_chunks = [
            AgentResponseChunk(text="Once ", done=False),
            AgentResponseChunk(text="upon a ", done=False),
            AgentResponseChunk(text="time, ", done=False),
            AgentResponseChunk(text="there was AI.", done=False),
            AgentResponseChunk(text="", done=True)
        ]
        
        with patch('agent.stream_agent_reply', new_callable=AsyncMock) as mock_stream:
            mock_stream.return_value = mock_chunks
            
            # When making streaming request
            client = TestClient(app)
            response = client.post("/api/chat/stream", json=chat_request)
            
            # Then should receive multiple chunks
            check.equal(response.status_code, 200)
            
            # Parse SSE response
            lines = response.text.strip().split('\n')
            data_lines = [line for line in lines if line.startswith('data: ')]
            
            # Should have multiple data chunks
            check.greater(len(data_lines), 1, "Should receive multiple chunks")
            
            # Verify chunk structure
            for line in data_lines:
                data = json.loads(line[6:])  # Remove 'data: ' prefix
                check.is_in('text', data)
                check.is_in('done', data)
    
    @pytest.mark.asyncio
    async def test_streaming_error_handling(self):
        """Test streaming error handling."""
        # Given a request that will cause error
        chat_request = {
            "session_id": "test_session", 
            "messages": []
        }
        
        # When making request with empty messages
        client = TestClient(app)
        response = client.post("/api/chat/stream", json=chat_request)
        
        # Then should handle error gracefully
        check.equal(response.status_code, 200)  # Streaming starts even with errors
        
        # Parse response for error
        lines = response.text.strip().split('\n')
        data_lines = [line for line in lines if line.startswith('data: ')]
        
        check.greater_equal(len(data_lines), 1)
    
    @pytest.mark.asyncio
    async def test_streaming_timeout_handling(self):
        """Test streaming timeout scenarios."""
        with patch('agent.stream_agent_reply', new_callable=AsyncMock) as mock_stream:
            # Simulate timeout by making stream hang
            async def hanging_stream():
                await asyncio.sleep(10)  # Longer than test timeout
                yield AgentResponseChunk(text="Never arrives", done=True)
            
            mock_stream.return_value = hanging_stream()
            
            # When making request that times out
            client = TestClient(app)
            chat_request = {
                "session_id": "test_session",
                "messages": [{"role": "user", "content": "Test"}]
            }
            
            # This should be handled by the client timeout, not server error
            response = client.post("/api/chat/stream", json=chat_request, timeout=1.0)
            
            # The request might complete or timeout, but shouldn't crash
            check.is_true(response.status_code in [200, 500])


class TestStreamingContent:
    """Test streaming content validation."""
    
    @pytest.mark.asyncio 
    async def test_streaming_content_structure(self):
        """Test that streaming maintains proper SSE structure."""
        client = TestClient(app)
        
        chat_request = {
            "session_id": "test_session",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        with patch('agent.stream_agent_reply', new_callable=AsyncMock) as mock_stream:
            mock_stream.return_value = [
                AgentResponseChunk(text="Hello", done=False),
                AgentResponseChunk(text=" World", done=True)
            ]
            
            response = client.post("/api/chat/stream", json=chat_request)
            
            # Check SSE format
            lines = response.text.strip().split('\n')
            
            # Should have data lines and blank lines
            data_lines = [line for line in lines if line.startswith('data: ')]
            blank_lines = [line for line in lines if line == '']
            
            check.greater(len(data_lines), 0)
            check.greater(len(blank_lines), 0)
            
            # Each data line should be valid JSON
            for line in data_lines:
                data_content = line[6:]  # Remove 'data: '
                try:
                    data = json.loads(data_content)
                    check.is_instance(data, dict)
                    check.is_in('text', data)
                    check.is_in('done', data)
                except json.JSONDecodeError:
                    check.fail(f"Invalid JSON in SSE: {data_content}")


# Test the complete flow with real async operations
@pytest.mark.asyncio
async def test_complete_streaming_flow():
    """Test complete streaming flow with realistic delays."""
    client = TestClient(app)
    
    chat_request = {
        "session_id": "integration_test",
        "messages": [{"role": "user", "content": "Stream this response"}]
    }
    
    # Simulate realistic streaming with delays
    async def mock_stream_with_delays(session_id, messages):
        chunks = [
            "Thinking",
            " about",
            " your",
            " question",
            "."
        ]
        
        for chunk in chunks:
            await asyncio.sleep(0.01)  # Small delay between chunks
            yield AgentResponseChunk(text=chunk, done=False)
        
        yield AgentResponseChunk(text="", done=True)
    
    with patch('agent.stream_agent_reply', new_callable=AsyncMock) as mock_stream:
        mock_stream.side_effect = mock_stream_with_delays
        
        response = client.post("/api/chat/stream", json=chat_request)
        
        # Verify streaming works
        check.equal(response.status_code, 200)
        
        # Should receive multiple chunks
        content = response.text
        data_events = [line for line in content.split('\n') if line.startswith('data: ')]
        
        check.greater(len(data_events), 1, "Should receive multiple streaming events")