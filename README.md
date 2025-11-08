# RAG Document Chatbot (Agno Rag Agent)

A production-ready Retrieval-Augmented Generation chatbot built with FastAPI, Agno, and NiceGUI. Features real-time response streaming and PDF document processing for contextual question answering.

## Features

- **Real-time Streaming**: Token-by-token response streaming via Server-Sent Events
- **PDF Document Processing**: Upload and parse PDFs with semantic chunking
- **Multi-provider Support**: OpenAI, Gemini, and Groq integration
- **Professional UI**: Modern NiceGUI interface with auto-scrolling chat
- **Knowledge Management**: Monitor and manage document knowledge base
- **Async Operations**: Non-blocking UI with real-time status updates

## Architecture

```
Frontend (NiceGUI) ←→ FastAPI Backend ←→ Agno RAG Agent ←→ Vector Database
```

**Technology Stack**:
- **FastAPI**: Backend with SSE streaming endpoints
- **Agno**: RAG agent with knowledge base and session management
- **NiceGUI**: Reactive frontend with professional styling
- **LanceDB**: Vector database for document embeddings

## Installation

### Prerequisites

- Python 3.13+
- API key from OpenAI, Gemini, or Groq

### Setup

1. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the application**:
```bash
# Terminal 1 - Backend API
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend UI
python ui.py
```

Access the application at `http://localhost:8080`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/stream` | Stream chat responses (SSE) |
| `POST` | `/api/upload/pdf` | Upload and process PDF documents |
| `POST` | `/api/set_api_key` | Configure API keys dynamically |
| `GET` | `/api/knowledge/stats` | Get knowledge base statistics |
| `POST` | `/api/clear_knowledge` | Clear document knowledge |

## Testing

```bash
# Run complete test suite
pytest

# Run with coverage reporting
pytest --cov=.

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

**Test Structure**:
- **Unit Tests**: Agent configuration, PDF parsing, model validation
- **Integration Tests**: Streaming chunks, PDF upload flow, error handling
- **Test Data**: Sample PDFs in `tests/data/` directory

## Development

### Cursor Configuration

The project includes `.cursorrules` for optimized development workflow:

- Focus on simple, maintainable code
- Prefer Agno built-ins over custom implementations
- Write tests for all new functionality
- Use modern Python typing patterns

### Development Workflow

1. Start with test-driven development approach
2. Implement minimal working functionality
3. Validate streaming end-to-end
4. Add features incrementally with tests

## Deployment

### Local Development
```bash
uvicorn main:app --reload --port 8000
python ui.py
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Design Decisions

### Agno Framework
- Built-in RAG capabilities with multiple embedding backends
- Session management and history tracking
- Semantic chunking for optimal document processing
- Minimal configuration required

### FastAPI + NiceGUI
- FastAPI provides excellent SSE support for streaming
- NiceGUI offers reactive UI with minimal boilerplate
- Both support modern async/await patterns
- Easy integration and deployment

## Performance

- **Streaming**: Token-by-token responses with <100ms latency
- **PDF Processing**: Semantic chunking for optimal RAG performance
- **Async Operations**: Non-blocking UI during all operations
- **Memory Management**: Efficient vector database with configurable chunk sizes

## Contributing

1. Write tests for new functionality
2. Follow existing code style and patterns
3. Ensure all tests pass before submitting
4. Update documentation for new features

## License

MIT License - see LICENSE file for details.