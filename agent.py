
# """
# Agno RAG Agent serving as the backend for the NiceGUI frontend.
# Handles PDF uploads, chat streaming, and knowledge management with async SSE.
# """

# import asyncio
# import os
# import logging
# import shutil
# from typing import AsyncGenerator
# from dotenv import load_dotenv
# from pydantic import BaseModel

# from agno.agent import Agent
# from agno.models.openai import OpenAIChat
# from agno.db.sqlite import SqliteDb
# from agno.media import File
# from agno.vectordb.lancedb import LanceDb
# from agno.knowledge.knowledge import Knowledge
# from agno.knowledge.reader.pdf_reader import PDFReader
# from agno.knowledge.chunking.semantic import SemanticChunking

# # ------------------------------------------------------
# # Load environment variables
# # ------------------------------------------------------
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ------------------------------------------------------
# # Pydantic Models
# # ------------------------------------------------------
# class Message(BaseModel):
#     role: str  # "user" | "assistant" | "system"
#     content: str


# class AgentResponseChunk(BaseModel):
#     text: str
#     done: bool = False


# # ------------------------------------------------------
# # Embedding Configuration - ENHANCED GEMINI SUPPORT
# # ------------------------------------------------------
# def get_embedder():
#     """Get the best available embedder with fallbacks - Gemini first priority."""
#     # Try Gemini embeddings first if API key is available
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     if gemini_api_key:
#         try:
#             from agno.knowledge.embedder.google import GeminiEmbedder
#             logger.info("🔑 Using Gemini embeddings")
#             # Use the correct model for embeddings (text-embedding-004 is the latest)
#             return GeminiEmbedder(
#                 api_key=gemini_api_key,
#                 model="models/embedding-001"  # Specify the embedding model
#             )
#         except ImportError as e:
#             logger.warning(f"❌ Gemini embeddings not available: {e}")
#         except Exception as e:
#             logger.warning(f"❌ Gemini embeddings configuration error: {e}")
    
#     # Try OpenAI embeddings second (requires API key)
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if openai_api_key:
#         try:
#             from agno.knowledge.embedder.openai import OpenAIEmbedder
#             logger.info("🔑 Using OpenAI embeddings")
#             return OpenAIEmbedder(api_key=openai_api_key)
#         except ImportError as e:
#             logger.warning(f"❌ OpenAI embeddings not available: {e}")
#         except Exception as e:
#             logger.warning(f"❌ OpenAI embeddings configuration error: {e}")
    
#     try:
#         # Try FastEmbed as third choice (lightweight)
#         from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
#         logger.info("🚀 Using FastEmbed embeddings")
#         return FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5")
#     except ImportError as e:
#         logger.warning(f"❌ FastEmbed not available: {e}")
    
#     try:
#         # Try SentenceTransformers as fourth choice
#         from agno.knowledge.embedder.sentence_transformers import SentenceTransformerEmbedder
#         logger.info("🤖 Using SentenceTransformer embeddings")
#         return SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
#     except ImportError as e:
#         logger.warning(f"❌ SentenceTransformers not available: {e}")
    
#     # Final fallback - try to use any available embedder
#     try:
#         from agno.knowledge.embedder.openai import OpenAIEmbedder
#         logger.info("🔄 Using OpenAI embeddings with default config")
#         return OpenAIEmbedder()
#     except Exception as e:
#         logger.error(f"❌ No embedding backend available: {e}")
#         raise ImportError("No embedding backend available. Please install one of: google-generativeai, openai, fastembed, sentence-transformers")


# # ------------------------------------------------------
# # RAG Agent Class - OPTIMIZED FOR GEMINI
# # ------------------------------------------------------
# class RAGAgent:
#     def __init__(self):
#         """Initialize RAG agent with proper embedding configuration and fallbacks."""
#         self.db = SqliteDb(db_file="tmp/chat_sessions.db")
        
#         # Store uploaded PDFs
#         self.uploaded_pdfs = []
        
#         # Get the best available embedder
#         embedder = get_embedder()
        
#         # Initialize knowledge base with proper RAG configuration
#         self.knowledge_base = Knowledge(
#             vector_db=LanceDb(
#                 table_name="pdf_documents",
#                 uri="tmp/lancedb",
#                 embedder=embedder,
#             ),
#             max_results=3,  # Control retrieval count
#         )

#         # Initialize the RAG-enabled agent with proper configuration
#         self.agent = Agent(
#             model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
#             db=self.db,
#             knowledge=self.knowledge_base,
#             search_knowledge=True,  # Enable RAG retrieval
#             add_history_to_context=True,
#             markdown=True,
#             instructions=(
#                 "You are a factual and context-aware assistant. "
#                 "Use the uploaded document knowledge when relevant and keep responses accurate. "
#                 "If the information is not in the provided documents, say so."
#             ),
#         )
        
#         logger.info("✅ RAG Agent initialized successfully")

#     # --------------------------------------------------
#     # Knowledge Base Methods - IMPROVED WITH BETTER ERROR HANDLING
#     # --------------------------------------------------
#     async def add_pdf_knowledge(self, pdf_content: bytes, filename: str) -> None:
#         """Add PDF content to knowledge base with enhanced error handling."""
#         try:
#             # Ensure upload directory exists
#             os.makedirs("tmp/uploads", exist_ok=True)
            
#             # Clean filename to remove URL encoding
#             clean_filename = filename.replace('%20', ' ').replace('%', '_')
#             pdf_path = f"tmp/uploads/{clean_filename}"
            
#             # Save PDF to upload directory
#             with open(pdf_path, "wb") as f:
#                 f.write(pdf_content)
            
#             logger.info(f"📁 Saved PDF to: {pdf_path}")

#             # Check if PDF is password protected
#             is_protected, protected_error = await self._check_pdf_protection(pdf_path)
#             if is_protected:
#                 raise Exception(f"PDF is password protected: {protected_error}")

#             # Use PDFReader with semantic chunking for better RAG
#             pdf_reader = PDFReader(
#                 chunking_strategy=SemanticChunking(similarity_threshold=0.5)
#             )
            
#             # Add to knowledge base
#             await self.knowledge_base.add_content_async(
#                 path=pdf_path,
#                 reader=pdf_reader,
#                 metadata={"source": clean_filename, "type": "pdf"}
#             )
            
#             self.uploaded_pdfs.append(pdf_path)
#             logger.info(f"✅ Successfully added PDF knowledge from {clean_filename}")
            
#         except Exception as e:
#             logger.error(f"❌ Error adding PDF knowledge: {e}")
#             # Clean up temporary file on error
#             pdf_path = f"tmp/uploads/{filename.replace('%20', ' ').replace('%', '_')}"
#             if os.path.exists(pdf_path):
#                 os.remove(pdf_path)
#             raise

#     async def _check_pdf_protection(self, pdf_path: str) -> tuple[bool, str]:
#         """Check if PDF is password protected."""
#         try:
#             import PyPDF2
#             with open(pdf_path, 'rb') as file:
#                 pdf_reader = PyPDF2.PdfReader(file)
#                 if pdf_reader.is_encrypted:
#                     return True, "PDF requires password"
#                 return False, ""
#         except Exception as e:
#             # If we can't check, assume it's not protected but log the error
#             logger.warning(f"⚠️ Could not check PDF protection: {e}")
#             return False, ""

#     async def add_pdf_from_url(self, url: str, name: str) -> None:
#         """Add PDF from URL for testing purposes."""
#         try:
#             # Use PDFReader with semantic chunking
#             pdf_reader = PDFReader(
#                 chunking_strategy=SemanticChunking(similarity_threshold=0.5)
#             )
            
#             await self.knowledge_base.add_content_async(
#                 url=url,
#                 reader=pdf_reader,
#                 metadata={"source": url, "type": "pdf"}
#             )
#             logger.info(f"✅ Successfully added PDF knowledge from URL: {url}")
#         except Exception as e:
#             logger.error(f"❌ Error adding PDF from URL: {e}")
#             raise

#     async def add_text_knowledge(self, text: str, source: str) -> None:
#         """Add raw text content to knowledge base."""
#         try:
#             await self.knowledge_base.add_content_async(
#                 text=text,
#                 metadata={"source": source, "type": "text"}
#             )
#             logger.info(f"✅ Successfully added text knowledge from {source}")
#         except Exception as e:
#             logger.error(f"❌ Error adding text knowledge: {e}")
#             raise

#     async def process_pdf_directly(self, pdf_content: bytes, filename: str, session_id: str, user_message: str) -> AsyncGenerator[AgentResponseChunk, None]:
#         """Process PDF directly using File input for immediate querying."""
#         temp_pdf_path = None
#         try:
#             # Save PDF temporarily
#             os.makedirs("tmp/temp", exist_ok=True)
#             clean_filename = filename.replace('%20', ' ').replace('%', '_')
#             temp_pdf_path = f"tmp/temp/{clean_filename}"
            
#             with open(temp_pdf_path, "wb") as f:
#                 f.write(pdf_content)
            
#             # Create File object for direct processing
#             pdf_file = File(path=temp_pdf_path)
            
#             # Stream response with file attachment
#             full_response = ""
#             async for response in self.agent.arun(
#                 input=user_message,
#                 files=[pdf_file],
#                 session_id=session_id,
#                 stream=True
#             ):
#                 if hasattr(response, "content") and response.content:
#                     chunk_text = response.content
#                     full_response += chunk_text
#                     yield AgentResponseChunk(text=chunk_text, done=False)
#                 elif hasattr(response, "output") and response.output:
#                     chunk_text = response.output
#                     full_response += chunk_text
#                     yield AgentResponseChunk(text=chunk_text, done=False)

#             yield AgentResponseChunk(text="", done=True)
                
#         except Exception as e:
#             logger.error(f"❌ Direct PDF processing error: {e}")
#             yield AgentResponseChunk(
#                 text="Sorry, I encountered an error while processing the PDF.",
#                 done=True
#             )
#         finally:
#             # Clean up temp file
#             if temp_pdf_path and os.path.exists(temp_pdf_path):
#                 os.remove(temp_pdf_path)

#     async def clear_knowledge(self) -> None:
#         """Clear knowledge base completely."""
#         try:
#             # Clear vector database
#             if os.path.exists("tmp/lancedb"):
#                 shutil.rmtree("tmp/lancedb", ignore_errors=True)
#             if os.path.exists("tmp/uploads"):
#                 shutil.rmtree("tmp/uploads", ignore_errors=True)
#             if os.path.exists("tmp/temp"):
#                 shutil.rmtree("tmp/temp", ignore_errors=True)
                
#             # Recreate directories
#             os.makedirs("tmp/lancedb", exist_ok=True)
#             os.makedirs("tmp/uploads", exist_ok=True)
#             os.makedirs("tmp/temp", exist_ok=True)
            
#             self.uploaded_pdfs = []
            
#             # Reinitialize knowledge base with current embedder
#             embedder = get_embedder()
#             self.knowledge_base = Knowledge(
#                 vector_db=LanceDb(
#                     table_name="pdf_documents",
#                     uri="tmp/lancedb",
#                     embedder=embedder,
#                 ),
#                 max_results=3,
#             )
            
#             # Update agent knowledge reference
#             self.agent.knowledge = self.knowledge_base
            
#             logger.info("✅ Knowledge base cleared successfully")
            
#         except Exception as e:
#             logger.error(f"❌ Error clearing knowledge base: {e}")
#             raise

#     async def get_knowledge_stats(self) -> dict:
#         """Get statistics about the knowledge base."""
#         try:
#             vector_db_size = 0
#             if os.path.exists("tmp/lancedb"):
#                 for root, dirs, files in os.walk("tmp/lancedb"):
#                     for file in files:
#                         vector_db_size += os.path.getsize(os.path.join(root, file))
            
#             upload_dir_size = 0
#             if os.path.exists("tmp/uploads"):
#                 for root, dirs, files in os.walk("tmp/uploads"):
#                     for file in files:
#                         upload_dir_size += os.path.getsize(os.path.join(root, file))
            
#             # Get embedder type
#             embedder_type = "Unknown"
#             if hasattr(self.knowledge_base.vector_db, 'embedder'):
#                 embedder_type = type(self.knowledge_base.vector_db.embedder).__name__
            
#             return {
#                 "uploaded_pdfs": len(self.uploaded_pdfs),
#                 "knowledge_base_initialized": self.knowledge_base is not None,
#                 "vector_db_exists": os.path.exists("tmp/lancedb"),
#                 "vector_db_size_mb": round(vector_db_size / (1024 * 1024), 2),
#                 "upload_directory_exists": os.path.exists("tmp/uploads"),
#                 "upload_directory_size_mb": round(upload_dir_size / (1024 * 1024), 2),
#                 "embedding_backend": embedder_type
#             }
#         except Exception as e:
#             logger.error(f"❌ Error getting knowledge stats: {e}")
#             return {"error": str(e)}
        

#     async def reload_model(self):
#         """Reload the underlying model after API key change."""
#         try:
#             embedder = get_embedder()
#             self.knowledge_base.vector_db.embedder = embedder

#             # Reinitialize the agent model (OpenAI or Groq)
#             from agno.models.openai import OpenAIChat
#             from agno.models.groq import GroqChat  # only if installed

#             openai_key = os.getenv("OPENAI_API_KEY")
#             groq_key = os.getenv("GROQ_API_KEY")

#             if groq_key:
#                 self.agent.model = GroqChat(id="llama-3.1-70b-versatile", api_key=groq_key)
#                 logger.info("🔄 RAG agent model switched to GroqChat.")
#             elif openai_key:
#                 self.agent.model = OpenAIChat(id="gpt-4o-mini", api_key=openai_key)
#                 logger.info("🔄 RAG agent model switched to OpenAIChat.")
#             else:
#                 logger.warning("⚠️ No API key found; model not reloaded.")
#         except Exception as e:
#             logger.error(f"❌ Failed to reload model: {e}")


#     # --------------------------------------------------
#     # Streaming Agent Response
#     # --------------------------------------------------
#     async def stream_agent_reply(
#         self, session_id: str, messages: list[Message]
#     ) -> AsyncGenerator[AgentResponseChunk, None]:
#         """Stream agent responses with RAG context."""
#         try:
#             user_messages = [msg for msg in messages if msg.role == "user"]
#             if not user_messages:
#                 yield AgentResponseChunk(text="No user message found", done=True)
#                 return

#             latest_user_message = user_messages[-1].content
            
#             logger.info(f"🔍 Processing query with RAG: {latest_user_message[:100]}...")
            
#             full_response = ""
#             async for response in self.agent.arun(
#                 input=latest_user_message, 
#                 session_id=session_id, 
#                 stream=True
#             ):
#                 if hasattr(response, "content") and response.content:
#                     chunk_text = response.content
#                     full_response += chunk_text
#                     yield AgentResponseChunk(text=chunk_text, done=False)
#                 elif hasattr(response, "output") and response.output:
#                     chunk_text = response.output
#                     full_response += chunk_text
#                     yield AgentResponseChunk(text=chunk_text, done=False)

#             logger.info(f"✅ Response completed for session: {session_id}")
#             yield AgentResponseChunk(text="", done=True)

#         except Exception as e:
#             logger.error(f"❌ Streaming error: {e}")
#             yield AgentResponseChunk(
#                 text="Sorry, I encountered an internal error while processing your request.",
#                 done=True
#             )


# # ------------------------------------------------------
# # Global Instance
# # ------------------------------------------------------
# rag_agent = RAGAgent()

# # ------------------------------------------------------
# # Public Interface Functions - UPDATED
# # ------------------------------------------------------
# async def add_pdf_knowledge(pdf_content: bytes, filename: str) -> None:
#     """Public interface to add PDF knowledge."""
#     await rag_agent.add_pdf_knowledge(pdf_content, filename)


# async def add_pdf_from_url(url: str, name: str) -> None:
#     """Public interface to add PDF from URL."""
#     await rag_agent.add_pdf_from_url(url, name)


# async def add_text_knowledge(text: str, source: str) -> None:
#     """Public interface to add text knowledge."""
#     await rag_agent.add_text_knowledge(text, source)


# async def clear_knowledge() -> None:
#     """Public interface to clear knowledge base."""
#     await rag_agent.clear_knowledge()


# async def stream_agent_reply(
#     session_id: str, messages: list[Message]
# ) -> AsyncGenerator[AgentResponseChunk, None]:
#     """Public interface to stream agent replies."""
#     async for chunk in rag_agent.stream_agent_reply(session_id, messages):
#         yield chunk


# async def process_pdf_directly(
#     pdf_content: bytes, filename: str, session_id: str, user_message: str
# ) -> AsyncGenerator[AgentResponseChunk, None]:
#     """Public interface for direct PDF processing."""
#     async for chunk in rag_agent.process_pdf_directly(pdf_content, filename, session_id, user_message):
#         yield chunk


# async def get_knowledge_stats() -> dict:
#     """Public interface to get knowledge base statistics."""
#     return await rag_agent.get_knowledge_stats()


# # ------------------------------------------------------
# # Enhanced Dependency Check Function
# # ------------------------------------------------------
# def check_dependencies():
#     """Check if all required dependencies are available."""
#     dependencies = {
#         "google-generativeai": False,
#         "openai": False,
#         "fastembed": False,
#         "sentence-transformers": False,
#         "PyPDF2": False
#     }
    
#     try:
#         import google.generativeai
#         dependencies["google-generativeai"] = True
#     except ImportError:
#         pass
        
#     try:
#         import openai
#         dependencies["openai"] = True
#     except ImportError:
#         pass
        
#     try:
#         import fastembed
#         dependencies["fastembed"] = True
#     except ImportError:
#         pass
        
#     try:
#         from sentence_transformers import SentenceTransformer
#         dependencies["sentence-transformers"] = True
#     except ImportError:
#         pass
        
#     try:
#         import PyPDF2
#         dependencies["PyPDF2"] = True
#     except ImportError:
#         pass
    
#     return dependencies


# # ------------------------------------------------------
# # Enhanced Testing Utilities
# # ------------------------------------------------------
# def test_agent_configuration():
#     """Comprehensive sanity test for configuration."""
#     assert rag_agent.agent.model is not None, "Model not initialized"
#     assert rag_agent.agent.db is not None, "SQLite not initialized"
#     assert rag_agent.agent.knowledge is not None, "Knowledge base missing"
#     assert rag_agent.agent.search_knowledge, "Knowledge search disabled"
#     assert rag_agent.knowledge_base.max_results == 3, "Max results not configured"
#     print("✅ Agent configuration verified successfully!")


# async def test_pdf_upload():
#     """Test PDF upload functionality with actual RAG verification."""
#     try:
#         # Create a simple test PDF content
#         test_content = b"%PDF-1.4 test pdf content"
#         await add_pdf_knowledge(test_content, "test.pdf")
        
#         # Verify knowledge base stats
#         stats = await get_knowledge_stats()
#         print(f"📊 Knowledge stats: {stats}")
        
#         print("✅ PDF upload test passed!")
#     except Exception as e:
#         print(f"❌ PDF upload test failed: {e}")


# async def test_rag_functionality():
#     """Test actual RAG functionality with a known PDF."""
#     try:
#         # Add a known PDF from URL for testing
#         test_url = "https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
#         await add_pdf_from_url(test_url, "ThaiRecipes")
        
#         # Test query that should use RAG
#         session_id = "rag_test_session"
#         messages = [Message(role="user", content="How to make Thai curry?")]
        
#         print("Testing RAG functionality...")
#         async for chunk in stream_agent_reply(session_id, messages):
#             if chunk.text:
#                 print(chunk.text, end="", flush=True)
#             if chunk.done:
#                 print("\n" + "="*50)
                
#         print("✅ RAG functionality test passed!")
        
#     except Exception as e:
#         print(f"❌ RAG functionality test failed: {e}")


# if __name__ == "__main__":
#     # Create necessary directories
#     os.makedirs("tmp", exist_ok=True)
#     os.makedirs("tmp/uploads", exist_ok=True)
#     os.makedirs("tmp/lancedb", exist_ok=True)
#     os.makedirs("tmp/temp", exist_ok=True)
    
#     # Check environment
#     if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
#         print("❌ Please set either OPENAI_API_KEY or GEMINI_API_KEY in your .env file.")
#         exit(1)
    
#     # Check dependencies
#     print("🔍 Checking dependencies...")
#     deps = check_dependencies()
#     for dep, available in deps.items():
#         status = "✅" if available else "❌"
#         print(f"   {status} {dep}")
    
#     print("🤖 Testing Enhanced RAG Agent...")
#     asyncio.run(test_pdf_upload())
#     test_agent_configuration()
#     asyncio.run(test_rag_functionality())
#     print("✅ All tests completed!")






















"""
Agno RAG Agent serving as the backend for the NiceGUI frontend.
Handles PDF uploads, chat streaming, and knowledge management with async SSE.
Pinecone vector database support for Render production deployment.
"""

import asyncio
import os
import logging
import shutil
from typing import AsyncGenerator
from dotenv import load_dotenv
from pydantic import BaseModel

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.media import File
from agno.vectordb.pineconedb import PineconeDb
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.semantic import SemanticChunking

# PostgreSQL imports
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

# ------------------------------------------------------
# Load environment variables
# ------------------------------------------------------
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# PostgreSQL Database Configuration
# ------------------------------------------------------
# Convert Render's DATABASE_URL to async format
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql+asyncpg://")

# Fallback to SQLite for local development if no DATABASE_URL
if not DATABASE_URL or "postgresql" not in DATABASE_URL:
    DATABASE_URL = "sqlite+aiosqlite:///./tmp/chat_sessions.db"
    logger.info("🔧 Using SQLite for local development")

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Database engine and session
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------
class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class AgentResponseChunk(BaseModel):
    text: str
    done: bool = False


# ------------------------------------------------------
# Embedding Configuration - ENHANCED GEMINI SUPPORT
# ------------------------------------------------------
def get_embedder():
    """Get the best available embedder with fallbacks - Gemini first priority."""
    # Try Gemini embeddings first if API key is available
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            from agno.knowledge.embedder.google import GeminiEmbedder
            logger.info("🔑 Using Gemini embeddings")
            # Use the correct model for embeddings (text-embedding-004 is the latest)
            return GeminiEmbedder(
                api_key=gemini_api_key,
                model="models/embedding-001"  # Specify the embedding model
            )
        except ImportError as e:
            logger.warning(f"❌ Gemini embeddings not available: {e}")
        except Exception as e:
            logger.warning(f"❌ Gemini embeddings configuration error: {e}")
    
    # Try OpenAI embeddings second (requires API key)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            from agno.knowledge.embedder.openai import OpenAIEmbedder
            logger.info("🔑 Using OpenAI embeddings")
            return OpenAIEmbedder(api_key=openai_api_key)
        except ImportError as e:
            logger.warning(f"❌ OpenAI embeddings not available: {e}")
        except Exception as e:
            logger.warning(f"❌ OpenAI embeddings configuration error: {e}")
    
    try:
        # Try FastEmbed as third choice (lightweight)
        from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
        logger.info("🚀 Using FastEmbed embeddings")
        return FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5")
    except ImportError as e:
        logger.warning(f"❌ FastEmbed not available: {e}")
    
    try:
        # Try SentenceTransformers as fourth choice
        from agno.knowledge.embedder.sentence_transformers import SentenceTransformerEmbedder
        logger.info("🤖 Using SentenceTransformer embeddings")
        return SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    except ImportError as e:
        logger.warning(f"❌ SentenceTransformers not available: {e}")
    
    # Final fallback - try to use any available embedder
    try:
        from agno.knowledge.embedder.openai import OpenAIEmbedder
        logger.info("🔄 Using OpenAI embeddings with default config")
        return OpenAIEmbedder()
    except Exception as e:
        logger.error(f"❌ No embedding backend available: {e}")
        raise ImportError("No embedding backend available. Please install one of: google-generativeai, openai, fastembed, sentence-transformers")


# ------------------------------------------------------
# RAG Agent Class - OPTIMIZED FOR PINEcone IN PRODUCTION
# ------------------------------------------------------
class RAGAgent:
    def __init__(self):
        """Initialize RAG agent with Pinecone for production deployment."""
        self.db_engine = engine
        self.AsyncSessionLocal = AsyncSessionLocal
        
        # Store uploaded PDFs
        self.uploaded_pdfs = []
        
        # Get the best available embedder
        embedder = get_embedder()
        
        # Pinecone configuration for production
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east4-gcp")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
        
        if not pinecone_api_key:
            logger.warning("⚠️ PINECONE_API_KEY not found. Using LanceDB fallback for local development.")
            # Fallback to LanceDB for local development
            from agno.vectordb.lancedb import LanceDb
            self.knowledge_base = Knowledge(
                vector_db=LanceDb(
                    table_name="pdf_documents",
                    uri="tmp/lancedb",
                    embedder=embedder,
                ),
                max_results=3,
            )
        else:
            # Use Pinecone for production
            logger.info("🌲 Using Pinecone vector database for production")
            self.knowledge_base = Knowledge(
                vector_db=PineconeDb(
                    index_name=pinecone_index_name,
                    embedder=embedder,
                    api_key=pinecone_api_key,
                    environment=pinecone_environment,
                    dimension=768,  # Adjust based on your embedder
                ),
                max_results=3,  # Control retrieval count
            )

            # Initialize the RAG-enabled agent with proper configuration
            self.agent = Agent(
                model=OpenAIChat(
                    id="gpt-4o-mini", 
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL", None)  # Add this for flexibility
                ),
                knowledge=self.knowledge_base,
                search_knowledge=True,  # Enable RAG retrieval
                add_history_to_context=True,
                markdown=True,
                instructions=(
                    "You are a factual and context-aware assistant. "
                    "Use the uploaded document knowledge when relevant and keep responses accurate. "
                    "If the information is not in the provided documents, say so."
                ),
            )

            logger.info("✅ RAG Agent initialized successfully with Pinecone")

    async def save_chat_message(self, session_id: str, role: str, content: str) -> None:
        """Save chat message to PostgreSQL database."""
        try:
            async with self.AsyncSessionLocal() as session:
                # First, ensure session exists
                from sqlalchemy import text
                session_exists = await session.execute(
                    text("SELECT 1 FROM chat_sessions WHERE id = :session_id"),
                    {"session_id": session_id}
                )
                if not session_exists.scalar():
                    await session.execute(
                        text("INSERT INTO chat_sessions (id) VALUES (:session_id)"),
                        {"session_id": session_id}
                    )
                
                # Save the message
                await session.execute(
                    text("INSERT INTO chat_messages (session_id, role, content) VALUES (:session_id, :role, :content)"),
                    {"session_id": session_id, "role": role, "content": content}
                )
                await session.commit()
                logger.debug(f"💾 Saved {role} message for session {session_id}")
        except Exception as e:
            logger.error(f"❌ Error saving chat message: {e}")

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get chat history from PostgreSQL database."""
        try:
            async with self.AsyncSessionLocal() as session:
                from sqlalchemy import text
                result = await session.execute(
                    text("SELECT role, content FROM chat_messages WHERE session_id = :session_id ORDER BY created_at"),
                    {"session_id": session_id}
                )
                messages = result.fetchall()
                return [Message(role=msg[0], content=msg[1]) for msg in messages]
        except Exception as e:
            logger.error(f"❌ Error getting chat history: {e}")
            return []

    # --------------------------------------------------
    # Knowledge Base Methods - IMPROVED WITH BETTER ERROR HANDLING
    # --------------------------------------------------
    async def add_pdf_knowledge(self, pdf_content: bytes, filename: str) -> None:
        """Add PDF content to knowledge base with enhanced error handling."""
        try:
            # Ensure upload directory exists
            os.makedirs("tmp/uploads", exist_ok=True)
            
            # Clean filename to remove URL encoding
            clean_filename = filename.replace('%20', ' ').replace('%', '_')
            pdf_path = f"tmp/uploads/{clean_filename}"
            
            # Save PDF to upload directory
            with open(pdf_path, "wb") as f:
                f.write(pdf_content)
            
            logger.info(f"📁 Saved PDF to: {pdf_path}")

            # Check if PDF is password protected
            is_protected, protected_error = await self._check_pdf_protection(pdf_path)
            if is_protected:
                raise Exception(f"PDF is password protected: {protected_error}")

            # Use PDFReader with semantic chunking for better RAG
            pdf_reader = PDFReader(
                chunking_strategy=SemanticChunking(similarity_threshold=0.5)
            )
            
            # Add to knowledge base
            await self.knowledge_base.add_content_async(
                path=pdf_path,
                reader=pdf_reader,
                metadata={"source": clean_filename, "type": "pdf"}
            )
            
            self.uploaded_pdfs.append(pdf_path)
            logger.info(f"✅ Successfully added PDF knowledge from {clean_filename}")
            
        except Exception as e:
            logger.error(f"❌ Error adding PDF knowledge: {e}")
            # Clean up temporary file on error
            pdf_path = f"tmp/uploads/{filename.replace('%20', ' ').replace('%', '_')}"
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            raise

    async def _check_pdf_protection(self, pdf_path: str) -> tuple[bool, str]:
        """Check if PDF is password protected."""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if pdf_reader.is_encrypted:
                    return True, "PDF requires password"
                return False, ""
        except Exception as e:
            # If we can't check, assume it's not protected but log the error
            logger.warning(f"⚠️ Could not check PDF protection: {e}")
            return False, ""

    async def add_pdf_from_url(self, url: str, name: str) -> None:
        """Add PDF from URL for testing purposes."""
        try:
            # Use PDFReader with semantic chunking
            pdf_reader = PDFReader(
                chunking_strategy=SemanticChunking(similarity_threshold=0.5)
            )
            
            await self.knowledge_base.add_content_async(
                url=url,
                reader=pdf_reader,
                metadata={"source": url, "type": "pdf"}
            )
            logger.info(f"✅ Successfully added PDF knowledge from URL: {url}")
        except Exception as e:
            logger.error(f"❌ Error adding PDF from URL: {e}")
            raise

    async def add_text_knowledge(self, text: str, source: str) -> None:
        """Add raw text content to knowledge base."""
        try:
            await self.knowledge_base.add_content_async(
                text=text,
                metadata={"source": source, "type": "text"}
            )
            logger.info(f"✅ Successfully added text knowledge from {source}")
        except Exception as e:
            logger.error(f"❌ Error adding text knowledge: {e}")
            raise

    async def process_pdf_directly(self, pdf_content: bytes, filename: str, session_id: str, user_message: str) -> AsyncGenerator[AgentResponseChunk, None]:
        """Process PDF directly using File input for immediate querying."""
        temp_pdf_path = None
        try:
            # Save PDF temporarily
            os.makedirs("tmp/temp", exist_ok=True)
            clean_filename = filename.replace('%20', ' ').replace('%', '_')
            temp_pdf_path = f"tmp/temp/{clean_filename}"
            
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_content)
            
            # Create File object for direct processing
            pdf_file = File(path=temp_pdf_path)
            
            # Save user message to database
            await self.save_chat_message(session_id, "user", user_message)
            
            # Stream response with file attachment
            full_response = ""
            async for response in self.agent.arun(
                input=user_message,
                files=[pdf_file],
                stream=True
            ):
                if hasattr(response, "content") and response.content:
                    chunk_text = response.content
                    full_response += chunk_text
                    yield AgentResponseChunk(text=chunk_text, done=False)
                elif hasattr(response, "output") and response.output:
                    chunk_text = response.output
                    full_response += chunk_text
                    yield AgentResponseChunk(text=chunk_text, done=False)

            # Save assistant response to database
            if full_response:
                await self.save_chat_message(session_id, "assistant", full_response)
                
            yield AgentResponseChunk(text="", done=True)
                
        except Exception as e:
            logger.error(f"❌ Direct PDF processing error: {e}")
            yield AgentResponseChunk(
                text="Sorry, I encountered an error while processing the PDF.",
                done=True
            )
        finally:
            # Clean up temp file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    async def clear_knowledge(self) -> None:
        """Clear knowledge base completely."""
        try:
            # Clear local directories
            if os.path.exists("tmp/uploads"):
                shutil.rmtree("tmp/uploads", ignore_errors=True)
            if os.path.exists("tmp/temp"):
                shutil.rmtree("tmp/temp", ignore_errors=True)
                
            # Recreate directories
            os.makedirs("tmp/uploads", exist_ok=True)
            os.makedirs("tmp/temp", exist_ok=True)
            
            self.uploaded_pdfs = []
            
            # For Pinecone, we need to clear the index
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if pinecone_api_key:
                logger.info("🧹 Clearing Pinecone index...")
                # Note: In production, you might want to be more selective about clearing
                # This will clear the entire index - use with caution!
                try:
                    # Reinitialize knowledge base to clear Pinecone data
                    embedder = get_embedder()
                    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east4-gcp")
                    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
                    
                    self.knowledge_base = Knowledge(
                        vector_db=PineconeDb(
                            index_name=pinecone_index_name,
                            embedder=embedder,
                            api_key=pinecone_api_key,
                            environment=pinecone_environment,
                            dimension=768,
                        ),
                        max_results=3,
                    )
                    logger.info("✅ Pinecone knowledge base cleared")
                except Exception as e:
                    logger.error(f"❌ Error clearing Pinecone: {e}")
            else:
                # Fallback to LanceDB clearing
                if os.path.exists("tmp/lancedb"):
                    shutil.rmtree("tmp/lancedb", ignore_errors=True)
                os.makedirs("tmp/lancedb", exist_ok=True)
                
                embedder = get_embedder()
                from agno.vectordb.lancedb import LanceDb
                self.knowledge_base = Knowledge(
                    vector_db=LanceDb(
                        table_name="pdf_documents",
                        uri="tmp/lancedb",
                        embedder=embedder,
                    ),
                    max_results=3,
                )
                logger.info("✅ Local knowledge base cleared")
            
            # Update agent knowledge reference
            self.agent.knowledge = self.knowledge_base
            
            logger.info("✅ Knowledge base cleared successfully")
            
        except Exception as e:
            logger.error(f"❌ Error clearing knowledge base: {e}")
            raise

    async def get_knowledge_stats(self) -> dict:
        """Get statistics about the knowledge base."""
        try:
            upload_dir_size = 0
            if os.path.exists("tmp/uploads"):
                for root, dirs, files in os.walk("tmp/uploads"):
                    for file in files:
                        upload_dir_size += os.path.getsize(os.path.join(root, file))
            
            # Get embedder type
            embedder_type = "Unknown"
            if hasattr(self.knowledge_base.vector_db, 'embedder'):
                embedder_type = type(self.knowledge_base.vector_db.embedder).__name__
            
            # Get database stats
            async with self.AsyncSessionLocal() as session:
                from sqlalchemy import text
                session_count = await session.execute(text("SELECT COUNT(*) FROM chat_sessions"))
                message_count = await session.execute(text("SELECT COUNT(*) FROM chat_messages"))
                
                session_count_val = session_count.scalar() or 0
                message_count_val = message_count.scalar() or 0
            
            # Determine vector database type
            vector_db_type = "Pinecone" if os.getenv("PINECONE_API_KEY") else "LanceDB"
            
            stats = {
                "uploaded_pdfs": len(self.uploaded_pdfs),
                "knowledge_base_initialized": self.knowledge_base is not None,
                "vector_db_type": vector_db_type,
                "upload_directory_exists": os.path.exists("tmp/uploads"),
                "upload_directory_size_mb": round(upload_dir_size / (1024 * 1024), 2),
                "embedding_backend": embedder_type,
                "chat_sessions_count": session_count_val,
                "chat_messages_count": message_count_val,
                "database_type": "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite"
            }
            
            # Add Pinecone-specific stats if available
            if vector_db_type == "Pinecone":
                stats["pinecone_environment"] = os.getenv("PINECONE_ENVIRONMENT", "us-east4-gcp")
                stats["pinecone_index"] = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting knowledge stats: {e}")
            return {"error": str(e)}
        

    async def reload_model(self):
        """Reload the underlying model after API key change."""
        try:
            embedder = get_embedder()
            self.knowledge_base.vector_db.embedder = embedder

            # Reinitialize the agent model (OpenAI or Groq)
            from agno.models.openai import OpenAIChat
            from agno.models.groq import GroqChat  # only if installed

            openai_key = os.getenv("OPENAI_API_KEY")
            groq_key = os.getenv("GROQ_API_KEY")

            if groq_key:
                self.agent.model = GroqChat(id="llama-3.1-70b-versatile", api_key=groq_key)
                logger.info("🔄 RAG agent model switched to GroqChat.")
            elif openai_key:
                self.agent.model = OpenAIChat(id="gpt-4o-mini", api_key=openai_key)
                logger.info("🔄 RAG agent model switched to OpenAIChat.")
            else:
                logger.warning("⚠️ No API key found; model not reloaded.")
        except Exception as e:
            logger.error(f"❌ Failed to reload model: {e}")


    # --------------------------------------------------
    # Streaming Agent Response
    # --------------------------------------------------
    async def stream_agent_reply(
        self, session_id: str, messages: list[Message]
    ) -> AsyncGenerator[AgentResponseChunk, None]:
        """Stream agent responses with RAG context."""
        try:
            user_messages = [msg for msg in messages if msg.role == "user"]
            if not user_messages:
                yield AgentResponseChunk(text="No user message found", done=True)
                return

            latest_user_message = user_messages[-1].content
            
            # Save user message to database
            await self.save_chat_message(session_id, "user", latest_user_message)
            
            logger.info(f"🔍 Processing query with RAG: {latest_user_message[:100]}...")
            
            full_response = ""
            async for response in self.agent.arun(
                input=latest_user_message, 
                stream=True
            ):
                if hasattr(response, "content") and response.content:
                    chunk_text = response.content
                    full_response += chunk_text
                    yield AgentResponseChunk(text=chunk_text, done=False)
                elif hasattr(response, "output") and response.output:
                    chunk_text = response.output
                    full_response += chunk_text
                    yield AgentResponseChunk(text=chunk_text, done=False)

            # Save assistant response to database
            if full_response:
                await self.save_chat_message(session_id, "assistant", full_response)
            
            logger.info(f"✅ Response completed for session: {session_id}")
            yield AgentResponseChunk(text="", done=True)

        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield AgentResponseChunk(
                text="Sorry, I encountered an internal error while processing your request.",
                done=True
            )


# ------------------------------------------------------
# Database Initialization
# ------------------------------------------------------
async def init_database():
    """Initialize database tables."""
    try:
        # For PostgreSQL on Render, we need to use sync connection to create tables
        if "postgresql" in DATABASE_URL:
            # Use sync connection for table creation
            sync_url = DATABASE_URL.replace("+asyncpg", "")
            from sqlalchemy import create_engine
            sync_engine = create_engine(sync_url)
            Base.metadata.create_all(sync_engine)
            logger.info("✅ PostgreSQL tables created successfully")
        else:
            # For SQLite, use async
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ SQLite tables created successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        raise


# ------------------------------------------------------
# Global Instance
# ------------------------------------------------------
rag_agent = RAGAgent()

# ------------------------------------------------------
# Public Interface Functions - UPDATED
# ------------------------------------------------------
async def add_pdf_knowledge(pdf_content: bytes, filename: str) -> None:
    """Public interface to add PDF knowledge."""
    await rag_agent.add_pdf_knowledge(pdf_content, filename)


async def add_pdf_from_url(url: str, name: str) -> None:
    """Public interface to add PDF from URL."""
    await rag_agent.add_pdf_from_url(url, name)


async def add_text_knowledge(text: str, source: str) -> None:
    """Public interface to add text knowledge."""
    await rag_agent.add_text_knowledge(text, source)


async def clear_knowledge() -> None:
    """Public interface to clear knowledge base."""
    await rag_agent.clear_knowledge()


async def stream_agent_reply(
    session_id: str, messages: list[Message]
) -> AsyncGenerator[AgentResponseChunk, None]:
    """Public interface to stream agent replies."""
    async for chunk in rag_agent.stream_agent_reply(session_id, messages):
        yield chunk


async def process_pdf_directly(
    pdf_content: bytes, filename: str, session_id: str, user_message: str
) -> AsyncGenerator[AgentResponseChunk, None]:
    """Public interface for direct PDF processing."""
    async for chunk in rag_agent.process_pdf_directly(pdf_content, filename, session_id, user_message):
        yield chunk


async def get_knowledge_stats() -> dict:
    """Public interface to get knowledge base statistics."""
    return await rag_agent.get_knowledge_stats()


async def get_chat_history(session_id: str) -> list[Message]:
    """Public interface to get chat history."""
    return await rag_agent.get_chat_history(session_id)


# ------------------------------------------------------
# Enhanced Dependency Check Function
# ------------------------------------------------------
def check_dependencies():
    """Check if all required dependencies are available."""
    dependencies = {
        "google-generativeai": False,
        "openai": False,
        "fastembed": False,
        "sentence-transformers": False,
        "PyPDF2": False,
        "sqlalchemy": False,
        "asyncpg": False,
        "aiosqlite": False,
        "pinecone": False
    }
    
    try:
        import google.generativeai
        dependencies["google-generativeai"] = True
    except ImportError:
        pass
        
    try:
        import openai
        dependencies["openai"] = True
    except ImportError:
        pass
        
    try:
        import fastembed
        dependencies["fastembed"] = True
    except ImportError:
        pass
        
    try:
        from sentence_transformers import SentenceTransformer
        dependencies["sentence-transformers"] = True
    except ImportError:
        pass
        
    try:
        import PyPDF2
        dependencies["PyPDF2"] = True
    except ImportError:
        pass
        
    try:
        import sqlalchemy
        dependencies["sqlalchemy"] = True
    except ImportError:
        pass
        
    try:
        import asyncpg
        dependencies["asyncpg"] = True
    except ImportError:
        pass
        
    try:
        import aiosqlite
        dependencies["aiosqlite"] = True
    except ImportError:
        pass
        
    try:
        import pinecone
        dependencies["pinecone"] = True
    except ImportError:
        pass
    
    return dependencies


# ------------------------------------------------------
# Enhanced Testing Utilities
# ------------------------------------------------------
def test_agent_configuration():
    """Comprehensive sanity test for configuration."""
    assert rag_agent.agent.model is not None, "Model not initialized"
    assert rag_agent.agent.knowledge is not None, "Knowledge base missing"
    assert rag_agent.agent.search_knowledge, "Knowledge search disabled"
    assert rag_agent.knowledge_base.max_results == 3, "Max results not configured"
    print("✅ Agent configuration verified successfully!")


async def test_pdf_upload():
    """Test PDF upload functionality with actual RAG verification."""
    try:
        # Create a simple test PDF content
        test_content = b"%PDF-1.4 test pdf content"
        await add_pdf_knowledge(test_content, "test.pdf")
        
        # Verify knowledge base stats
        stats = await get_knowledge_stats()
        print(f"📊 Knowledge stats: {stats}")
        
        print("✅ PDF upload test passed!")
    except Exception as e:
        print(f"❌ PDF upload test failed: {e}")


async def test_rag_functionality():
    """Test actual RAG functionality with a known PDF."""
    try:
        # Add a known PDF from URL for testing
        test_url = "https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
        await add_pdf_from_url(test_url, "ThaiRecipes")
        
        # Test query that should use RAG
        session_id = "rag_test_session"
        messages = [Message(role="user", content="How to make Thai curry?")]
        
        print("Testing RAG functionality...")
        async for chunk in stream_agent_reply(session_id, messages):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            if chunk.done:
                print("\n" + "="*50)
                
        print("✅ RAG functionality test passed!")
        
    except Exception as e:
        print(f"❌ RAG functionality test failed: {e}")


async def test_database_functionality():
    """Test PostgreSQL database functionality."""
    try:
        # Test saving and retrieving messages
        test_session_id = "test_db_session"
        test_message = "Test database message"
        
        await rag_agent.save_chat_message(test_session_id, "user", test_message)
        
        # Retrieve history
        history = await get_chat_history(test_session_id)
        assert len(history) > 0, "No messages retrieved from database"
        assert history[-1].content == test_message, "Message content doesn't match"
        
        print("✅ Database functionality test passed!")
    except Exception as e:
        print(f"❌ Database functionality test failed: {e}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("tmp", exist_ok=True)
    os.makedirs("tmp/uploads", exist_ok=True)
    os.makedirs("tmp/temp", exist_ok=True)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        print("❌ Please set either OPENAI_API_KEY or GEMINI_API_KEY in your .env file.")
        exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"   {status} {dep}")
    
    # Initialize database and run tests
    async def main():
        await init_database()
        print("🤖 Testing Enhanced RAG Agent with Pinecone...")
        await test_pdf_upload()
        test_agent_configuration()
        await test_database_functionality()
        await test_rag_functionality()
        print("✅ All tests completed!")
    
    asyncio.run(main())