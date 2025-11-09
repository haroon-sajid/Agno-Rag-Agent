"""
FastAPI backend serving as a bridge between NiceGUI frontend and Agno RAG agent.
Handles PDF uploads, chat streaming, and knowledge management with async SSE.
"""

import os
import uuid
import logging
import json
import time
from typing import AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException, status, APIRouter, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import asyncio

from agent import (
    add_pdf_knowledge, 
    clear_knowledge, 
    stream_agent_reply, 
    get_knowledge_stats,
    Message
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create API router with /api prefix
api_router = APIRouter(prefix="/api")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    session_id: str
    messages: list[Message]

class UploadResponse(BaseModel):
    message: str
    filename: str
    file_size: int

class ErrorResponse(BaseModel):
    error: str

class KnowledgeStatsResponse(BaseModel):
    uploaded_pdfs: int
    knowledge_base_initialized: bool
    vector_db_exists: bool

@api_router.post("/chat/stream", response_class=StreamingResponse)
async def stream_chat_response(chat_request: ChatRequest):
    """Stream assistant responses token-by-token using Server-Sent Events"""
    try:
        logger.info(f"Starting chat stream for session: {chat_request.session_id}")
        
        async def generate_sse() -> AsyncGenerator[str, None]:
            """Generate Server-Sent Events from agent stream"""
            try:
                async for chunk in stream_agent_reply(
                    chat_request.session_id, 
                    chat_request.messages
                ):
                    # Format as SSE data with proper JSON serialization
                    chunk_data = {
                        "text": chunk.text,
                        "done": chunk.done
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)
                    
                    # Break if done
                    if chunk.done:
                        break
                        
            except asyncio.CancelledError:
                logger.info("SSE stream cancelled by client")
                raise
            except Exception as e:
                logger.error(f"Error in SSE generation: {e}")
                error_chunk = {"text": f"Error generating response: {str(e)}", "done": True}
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "X-Accel-Buffering": "no",  # Important for SSE
            }
        )
        
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start chat stream"
        )

@api_router.post("/upload/pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF file into agent's knowledge base using Agno's PDFKnowledgeBase"""
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Read file content
        contents = await file.read()
        
        # Validate file size (10MB max)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size must be less than 10MB"
            )
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        
        logger.info(f"Processing PDF upload: {file.filename} ({len(contents)} bytes)")
        
        # Add to knowledge base using Agno's PDFKnowledgeBase with semantic chunking
        await add_pdf_knowledge(contents, file.filename)
        
        return UploadResponse(
            message=f"Successfully processed and added {file.filename} to knowledge base with semantic chunking",
            filename=file.filename,
            file_size=len(contents)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during PDF processing: {str(e)}"
        )

@api_router.post("/clear_knowledge")
async def clear_knowledge_endpoint():
    """Clear the agent's knowledge base"""
    try:
        logger.info("Clearing knowledge base")
        await clear_knowledge()
        return {"message": "Knowledge base cleared successfully"}
    except Exception as e:
        logger.error(f"Knowledge clearance error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear knowledge base"
        )

@api_router.get("/knowledge/stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats_endpoint():
    """Get knowledge base statistics."""
    try:
        stats = await get_knowledge_stats()
        return KnowledgeStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Knowledge stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get knowledge statistics"
        )



@api_router.post("/set_api_key")
async def set_api_key(
    api_key: str = Body(..., embed=True),
    provider: str = Body("openai", embed=True)
):
    """
    Dynamically set the API key for OpenAI or Groq.
    This allows the user to input their key from the UI without restarting the backend.
    """
    try:
        if not api_key.strip():
            raise HTTPException(status_code=400, detail="API key cannot be empty.")

        # Detect provider and set key in environment
        provider = provider.lower().strip()
        if provider == "groq":
            os.environ["GROQ_API_KEY"] = api_key
            # Remove OpenAI key if set previously
            os.environ.pop("OPENAI_API_KEY", None)
            logger.info("🔑 GROQ API key set successfully.")
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            # Remove Groq key if set previously
            os.environ.pop("GROQ_API_KEY", None)
            logger.info("🔑 OpenAI API key set successfully.")

        # Optional: reload the agent model dynamically (only if you add a function for it)
        # from agent import rag_agent
        # await rag_agent.reload_model()

        return {"message": f"{provider.upper()} API key set successfully."}
    
    except Exception as e:
        logger.error(f"❌ Error setting API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to set API key.")


@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Chatbot API"}

# Create main FastAPI app
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Add GZip middleware for better performance
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Enable CORS for the integrated app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include the API router with /api prefix
app.include_router(api_router)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)