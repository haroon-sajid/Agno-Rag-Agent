

"""
NiceGUI frontend for RAG Chatbot - PROFESSIONAL UI REDESIGN
"""

import uuid
import asyncio
import json
from typing import List
from nicegui import ui, events
import aiohttp
from pydantic import BaseModel
import os
from datetime import datetime

# Import your Message class - if it's in a different module, adjust the import
try:
    from agent import Message
except ImportError:
    # Fallback if Message class is not available
    class Message(BaseModel):
        role: str
        content: str

class ChatMessage(BaseModel):
    role: str
    content: str

# Global chatbot instance
chatbot_instance = None

class RAGChatbot:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.messages: List[ChatMessage] = []
        self.api_base = "http://localhost:8000/api"  
        self.is_streaming = False
        self.current_stream_content = ""
        self.session = None
        self.upload_status = None
        self.chat_messages = None
        self.status_label = None
        self.chat_input = None
        self.chat_history = []

    async def initialize_session(self):
        """Initialize aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=300)
            self.session = aiohttp.ClientSession(timeout=timeout)

    def init_ui(self):
        """Initialize the UI components with professional redesign and auto-scrolling chat"""
        # Full viewport height and clean background
        ui.query('body').style('padding: 0; margin: 0; height: 100vh; background-color: #f8fafc; overflow: hidden;')

        with ui.row().classes('w-full h-screen overflow-hidden'):
            # --- Left Sidebar ---
            with ui.column().classes('bg-white w-80 h-full border-r border-gray-200 shadow-sm flex flex-col'):
                # App Logo/Title Section
                with ui.column().classes('p-6 border-b border-gray-100'):
                    with ui.row().classes('items-center gap-3'):
                        ui.icon('smart_toy', color='primary', size='lg').classes('text-blue-600')
                        ui.label('RAG Assistant').classes('text-2xl font-bold text-gray-900')
                    ui.label('AI-Powered Document Chat').classes('text-sm text-gray-500 mt-1')

                # Action Buttons Section
                with ui.column().classes('p-4 gap-3 flex-none'):
                    ui.button('New Chat', on_click=self.clear_chat, icon='add_circle') \
                        .props('flat color=primary').classes('w-full justify-start h-10 font-medium rounded-lg')
                    ui.button('Upload PDF', icon='upload_file', on_click=self.trigger_upload) \
                        .props('flat color=primary').classes('w-full justify-start h-10 font-medium rounded-lg')
                    self.upload = ui.upload(
                        on_upload=self.handle_pdf_upload,
                        on_rejected=lambda: ui.notify('Please select a PDF file under 5MB', type='negative', position='top'),
                        auto_upload=True
                    ).props('accept=.pdf').classes('hidden')

                # API Key Input Section
                with ui.column().classes('p-4 gap-3 border-b border-gray-100'):
                    ui.label('🔑 API Key').classes('text-sm font-semibold text-gray-700')
                    with ui.column().classes('gap-2'):
                        with ui.input(
                            placeholder='Enter your OpenAI or Groq API key...',
                            password=True,
                            password_toggle_button=True
                        ).props('dense outlined rounded').classes('w-full') as self.api_key_input:
                            # Add key event handler using the correct NiceGUI approach
                            self.api_key_input.on('keydown.enter', self.handle_api_key_submit)
                        
                        with ui.row().classes('w-full gap-2'):
                            ui.button('Save Key', on_click=self.save_api_key, icon='key') \
                                .props('flat color=primary').classes('flex-1 h-9 font-medium rounded-lg')
                            ui.button('', on_click=self.clear_api_key, icon='clear') \
                                .props('flat color=grey-6').classes('h-9 w-9 rounded-lg')

                # Chat History Section
                with ui.column().classes('flex-grow p-4 overflow-hidden'):
                #     with ui.row().classes('items-center justify-between w-full mb-3'):
                #         ui.label('Recent Chats').classes('text-sm font-semibold text-gray-700 uppercase tracking-wide')
                #         ui.icon('history', size='sm').classes('text-gray-400')

                    self.history_container = ui.column().classes('w-full gap-2 overflow-y-auto flex-grow pr-1')
                    with self.history_container:
                        # ui.button('Document Analysis Session', icon='forum') \
                        #     .props('flat').classes('w-full justify-start text-gray-700 rounded-lg py-3 hover:bg-blue-50 transition-colors')
                        ui.button('Research Questions', icon='description') \
                            .props('flat').classes('w-full justify-start text-gray-700 rounded-lg py-3 hover:bg-blue-50 transition-colors')
                        ui.button('Technical Discussion', icon='chat') \
                            .props('flat').classes('w-full justify-start text-gray-700 rounded-lg py-3 hover:bg-blue-50 transition-colors')

                # Bottom Actions Section
                with ui.column().classes('p-4 border-t border-gray-100 gap-2 flex-none'):
                    ui.button('Knowledge Base', on_click=self.show_knowledge_stats, icon='dataset') \
                        .props('flat color=positive').classes('w-full justify-start h-10 font-medium rounded-lg')
                    ui.button('Clear Data', on_click=self.clear_knowledge, icon='delete_sweep') \
                        .props('flat color=negative').classes('w-full justify-start h-10 font-medium rounded-lg')

            # --- Main Chat Area ---
            with ui.element('div').classes(
                'relative flex flex-col flex-grow h-full bg-gradient-to-br from-gray-50 to-blue-50/20 overflow-hidden'
            ):
                # Scrollable chat messages area
                with ui.scroll_area().classes('flex-grow w-full overflow-y-auto') as self.scroll_area:
                    self.chat_messages = ui.column().classes(
                        'w-full max-w-4xl mx-auto p-8 gap-8 pb-32'
                    )

                # Fixed input bar (pinned slightly above bottom)
                with ui.element('div').classes(
                    'absolute bottom-5 left-0 w-full bg-white/90 backdrop-blur-sm border-t border-gray-200/50 p-6'
                ):
                    with ui.column().classes('w-full max-w-4xl mx-auto gap-4'):
                        # Status label
                        with ui.row().classes('justify-center'):
                            self.status_label = ui.label('Ready to chat').classes(
                                'text-xs text-gray-500 bg-white/80 px-3 py-1 rounded-full border border-gray-200'
                            )

                        # Input + Send row
                        with ui.row().classes('w-full gap-3 items-end'):
                            self.chat_input = ui.textarea(placeholder='Message RAG Assistant...') \
                                .props('outlined rounded autogrow rows=1') \
                                .classes('flex-grow bg-white border-0 shadow-sm') \
                                .style('--q-field-border-color: #e2e8f0;')

                            ui.button('Send', on_click=self.send_message, icon='send') \
                                .props('unelevated color=primary rounded') \
                                .classes('min-w-20 h-12 shadow-sm hover:shadow-md transition-shadow')

        # Auto-scroll helper method
        def scroll_to_bottom():
            try:
                # Use the correct NiceGUI API for scrolling
                self.scroll_area.scroll_to(percent=100)
            except Exception as e:
                print(f"Scroll error: {e}")

        self.scroll_to_bottom = scroll_to_bottom

        # Hook into chat updates to ensure both user and bot messages trigger scroll
        original_add_message = getattr(self, 'add_message', None)
        if original_add_message:
            def wrapped_add_message(*args, **kwargs):
                result = original_add_message(*args, **kwargs)
                self.scroll_to_bottom()
                return result
            self.add_message = wrapped_add_message

        # Show welcome message after a short delay
        ui.timer(0.1, lambda: self.show_welcome_message(), once=True)


    async def handle_api_key_submit(self):
        """Handle Enter key press in API key input"""
        await self.save_api_key()

    async def save_api_key(self):
        """Save API key to backend"""
        api_key = self.api_key_input.value.strip()
        
        if not api_key:
            ui.notify('Please enter an API key', type='warning', position='top')
            return

        # Determine provider based on key prefix
        provider = "openai"
        if api_key.startswith('gsk_'):
            provider = "groq"
        
        try:
            await self.initialize_session()
            
            payload = {
                "api_key": api_key,
                "provider": provider
            }
            
            async with self.session.post(
                f"{self.api_base}/set_api_key",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    ui.notify('✅ API Key saved successfully', type='positive', position='top')
                    # Clear the input field for security
                    self.api_key_input.value = ""
                else:
                    error_data = await response.json()
                    error_msg = error_data.get('detail', 'Unknown error')
                    ui.notify(f'❌ Failed to save API key: {error_msg}', type='negative', position='top')
                    
        except asyncio.TimeoutError:
            ui.notify('❌ Request timeout - please try again', type='negative', position='top')
        except aiohttp.ClientError as ex:
            ui.notify(f'❌ Network error: {str(ex)}', type='negative', position='top')
        except Exception as ex:
            ui.notify(f'❌ Unexpected error: {str(ex)}', type='negative', position='top')

    async def clear_api_key(self):
        """Clear the API key input field"""
        self.api_key_input.value = ""
        ui.notify('API key field cleared', type='info', position='top')

    def trigger_upload(self):
        """Trigger the hidden file upload"""
        self.upload.run_method('pickFiles')

    async def handle_pdf_upload(self, e: events.UploadEventArguments):
        """PDF upload handler with professional UI"""
        try:
            print(f"📤 Upload event received: {e}")
            
            if not e or not hasattr(e, 'file') or e.file is None:
                ui.notify('Upload error: No file selected', type='negative', position='top')
                return

            uploaded_file = e.file
            filename = uploaded_file.name
            
            print(f"📄 Processing file: {filename}")
            
            ui.notify(f'Processing {filename}...', type='info', position='top')

            # Validate file type
            if not filename.lower().endswith('.pdf'):
                ui.notify('Please select a PDF file', type='negative', position='top')
                return

            # Read file content
            try:
                file_content = await uploaded_file.read()
                print(f"📄 File read successfully: {len(file_content)} bytes")
            except Exception as read_error:
                ui.notify(f'Error reading file: {str(read_error)}', type='negative', position='top')
                return

            if len(file_content) == 0:
                ui.notify('Upload error: File is empty', type='negative', position='top')
                return

            if len(file_content) > 50 * 1024 * 1024:  # 50MB limit
                ui.notify('File too large (max 50MB)', type='negative', position='top')
                return

            print(f"📄 Uploading PDF to backend: {filename}, Size: {len(file_content)} bytes")

            # Initialize session if needed
            await self.initialize_session()

            # Prepare form data for FastAPI
            form_data = aiohttp.FormData()
            form_data.add_field(
                'file',
                file_content,
                filename=filename,
                content_type='application/pdf'
            )

            print(f"🌐 Sending request to {self.api_base}/upload/pdf")
            
            async with self.session.post(
                f"{self.api_base}/upload/pdf",
                data=form_data,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                
                print(f"📡 Server response: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    # Check if client still exists before notifying
                    try:
                        if ui.client:  # ensures there is an active client
                            ui.notify(
                                'PDF uploaded successfully!',
                                type='positive',
                                position='top'
                            )
                    except Exception as e:
                        print('Notification skipped, client disconnected:', e)

                                    
                    # Add success message to chat
                    await self.add_message("assistant", 
                        f"**PDF Successfully Uploaded!** 📄\n\n"
                        f"I've analyzed **{filename}** and added it to my knowledge base. "
                        f"You can now ask me questions about this document!\n\n"
                        f"*Try asking: 'What is this document about?' or 'Summarize the main points'*"
                    )
                    
                elif response.status == 413:
                    ui.notify('File too large for server processing', type='negative', position='top')
                else:
                    error_text = await response.text()
                    try:
                        error_data = json.loads(error_text)
                        error_detail = error_data.get('detail', f'HTTP {response.status}')
                    except:
                        error_detail = f'HTTP {response.status}: {error_text}'
                    ui.notify(f"Upload failed: {error_detail}", type='negative', position='top')
                    print(f"❌ Upload failed: {error_detail}")

        except asyncio.TimeoutError:
            ui.notify('Upload timeout - try with a smaller file', type='negative', position='top')
            print(f"❌ Upload timeout")
        except aiohttp.ClientError as ex:
            ui.notify('Network error - check server connection', type='negative', position='top')
            print(f"❌ Network error: {ex}")
        except Exception as ex:
            ui.notify('Upload failed due to unexpected error', type='negative', position='top')
            print(f"📛 Upload exception: {ex}")
            import traceback
            print(f"📛 Traceback: {traceback.format_exc()}")

    async def show_welcome_message(self):
        """Display welcome message"""
        await self.add_message("assistant", 
            "👋 **Welcome to RAG Document Chatbot!**\n\n"
            "**What you can do:**\n"
            "• 📄 Upload PDF documents using the upload button\n"  
            "• 💬 Ask questions about your uploaded documents\n"
            "• 🧠 Have general conversations with me\n"
            "• 🔍 Get answers based on document content\n\n"
            "**Get started by uploading a PDF or asking a question!**"
        )

    async def add_message(self, role: str, content: str):
        """Add message to chat display with professional styling"""
        if self.chat_messages is None:
            return
            
        self.messages.append(ChatMessage(role=role, content=content))
        timestamp = datetime.now().strftime("%H:%M")

        with self.chat_messages:
            if role == "user":
                with ui.row().classes('w-full justify-end'):
                    with ui.column().classes('max-w-[85%] gap-2'):
                        with ui.card().classes('bg-gradient-to-r from-blue-600 to-blue-500 text-white p-4 rounded-2xl rounded-br-none shadow-lg border-0'):
                            ui.markdown(content).classes('text-white text-sm leading-relaxed')
                        with ui.row().classes('justify-end items-center gap-2'):
                            ui.label(timestamp).classes('text-xs text-gray-500')
                            ui.icon('person', size='xs').classes('text-blue-500')
            else:
                with ui.row().classes('w-full justify-start'):
                    with ui.column().classes('max-w-[85%] gap-2'):
                        with ui.card().classes('bg-white p-4 rounded-2xl rounded-bl-none shadow-md border border-gray-100'):
                            ui.markdown(content).classes('text-gray-800 text-sm leading-relaxed')
                        with ui.row().classes('justify-start items-center gap-2'):
                            ui.icon('smart_toy', size='xs').classes('text-green-500')
                            ui.label(timestamp).classes('text-xs text-gray-500')

        # Scroll to bottom
        await asyncio.sleep(0.1)
        try:
            self.scroll_to_bottom()
        except Exception:
            pass

    async def send_message(self):
        """Send user message"""
        if self.chat_input is None or not self.chat_input.value.strip() or self.is_streaming:
            return

        user_text = self.chat_input.value.strip()
        self.chat_input.value = ""
        await self.add_message("user", user_text)

        # Convert to agent messages
        try:
            api_messages = []
            for msg in self.messages:
                if hasattr(msg, 'dict'):
                    api_messages.append(msg.dict())
                elif hasattr(msg, '__dict__'):
                    api_messages.append(msg.__dict__)
                else:
                    api_messages.append({"role": msg.role, "content": msg.content})
        except Exception as e:
            print(f"⚠️ Message conversion warning: {e}")
            api_messages = [{"role": msg.role, "content": msg.content} for msg in self.messages]

        print(f"📤 Sending {len(api_messages)} messages to backend")

        self.is_streaming = True
        self.current_stream_content = ""
        if self.status_label:
            self.status_label.set_text('Thinking...')

        try:
            # Create streaming message area
            if self.chat_messages:
                with self.chat_messages:
                    with ui.row().classes('w-full justify-start'):
                        with ui.column().classes('max-w-[85%] gap-2'):
                            self.streaming_card = ui.card().classes('bg-white p-4 rounded-2xl rounded-bl-none shadow-md border border-gray-100')
                            with self.streaming_card:
                                ui.markdown('').classes('text-gray-800 text-sm leading-relaxed').bind_content_from(self, 'current_stream_content')
            
            await self.stream_assistant_response(api_messages)
            
        except Exception as ex:
            ui.notify(f"Chat error: {str(ex)}", type='negative', position='top')
            await self.add_message("assistant", f"Sorry, I encountered an error: {str(ex)}")
        finally:
            self.is_streaming = False
            if self.status_label:
                self.status_label.set_text('Ready to chat')

    async def stream_assistant_response(self, api_messages):
        """Stream assistant response from backend"""
        chat_request = {
            "session_id": self.session_id, 
            "messages": api_messages
        }
        
        try:
            await self.initialize_session()
            
            print(f"🌐 Sending chat request to {self.api_base}/chat/stream")
            
            async with self.session.post(
                f"{self.api_base}/chat/stream", 
                json=chat_request,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                
                print(f"📡 Chat response status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"Server error {response.status}: {error_text}"
                    print(f"❌ {error_msg}")
                    raise Exception(error_msg)

                # Process Server-Sent Events
                buffer = ""
                async for chunk in response.content:
                    if chunk:
                        chunk_text = chunk.decode('utf-8')
                        buffer += chunk_text
                        
                        lines = buffer.split('\n')
                        buffer = lines[-1]
                        
                        for line in lines[:-1]:
                            line = line.strip()
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    text_chunk = data.get('text', '')
                                    done = data.get('done', False)
                                    
                                    if text_chunk:
                                        self.current_stream_content += text_chunk
                                        if self.status_label:
                                            self.status_label.set_text('Generating response...')
                                    
                                    if done:
                                        print("✅ Stream completed")
                                        break
                                        
                                except json.JSONDecodeError as e:
                                    print(f"❌ JSON decode error: {e}, line: {line}")
                                    continue
                                except Exception as e:
                                    print(f"❌ Error processing chunk: {e}")
                                    continue

            # Finalize the message
            if self.current_stream_content.strip():
                await self.add_message("assistant", self.current_stream_content.strip())
            else:
                await self.add_message("assistant", "I'm sorry, I didn't receive a response. Please try again.")
            
            # Clean up streaming UI
            if hasattr(self, 'streaming_card'):
                self.streaming_card.delete()
                
        except asyncio.TimeoutError:
            if hasattr(self, 'streaming_card'):
                self.streaming_card.delete()
            ui.notify("Request timeout - try again", type='warning', position='top')
            await self.add_message("assistant", "The request timed out. Please try again with a simpler question.")
            print("❌ Chat request timeout")
        except Exception as ex:
            if hasattr(self, 'streaming_card'):
                self.streaming_card.delete()
            ui.notify(f"Chat error: {str(ex)}", type='negative', position='top')
            await self.add_message("assistant", f"Sorry, I encountered an error while processing your request: {str(ex)}")
            print(f"❌ Chat stream error: {ex}")

    async def clear_chat(self):
        """Clear chat history"""
        self.messages.clear()
        if self.chat_messages:
            self.chat_messages.clear()
        await self.add_message("assistant", "🗑️ **Chat cleared!**\n\nI'm ready for new questions. Feel free to upload documents or ask me anything!")
        ui.notify("Chat cleared!", type='info', position='top')

    async def clear_knowledge(self):
        """Clear knowledge base"""
        try:
            await self.initialize_session()
            async with self.session.post(f"{self.api_base}/clear_knowledge") as response:
                if response.status == 200:
                    result = await response.json()
                    ui.notify("Knowledge base cleared!", type='positive', position='top')
                    await self.add_message("assistant", 
                        "🧹 **Document knowledge cleared!**\n\n"
                        "I've removed all uploaded documents from my memory. "
                        "You can still chat with me generally, or upload new PDFs to ask questions about."
                    )
                else:
                    error_data = await response.json()
                    ui.notify(f"Clear failed: {error_data.get('detail', 'Unknown error')}", type='negative', position='top')
        except Exception as ex:
            ui.notify(f"Clear error: {str(ex)}", type='negative', position='top')

    async def show_knowledge_stats(self):
        """Show knowledge base statistics"""
        try:
            await self.initialize_session()
            async with self.session.get(f"{self.api_base}/knowledge/stats") as response:
                if response.status == 200:
                    stats = await response.json()
                    
                    with ui.dialog() as dialog, ui.card().classes('p-6 max-w-md rounded-2xl shadow-xl'):
                        with ui.column().classes('gap-4'):
                            ui.label('📊 Knowledge Base Statistics').classes('text-xl font-bold text-gray-900')
                            
                            with ui.card().classes('bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-xl border-0'):
                                with ui.column().classes('gap-3'):
                                    with ui.row().classes('items-center gap-3'):
                                        ui.icon('description', color='primary').classes('text-blue-600')
                                        ui.label(f"Uploaded PDFs: {stats.get('uploaded_pdfs', 0)}").classes('font-semibold text-gray-800')
                                    
                                    with ui.row().classes('items-center gap-3'):
                                        ui.icon('storage', color='positive').classes('text-green-600')
                                        status = '✅ Initialized' if stats.get('knowledge_base_initialized') else '❌ Not initialized'
                                        ui.label(f"Knowledge Base: {status}").classes('font-semibold text-gray-800')
                                    
                                    with ui.row().classes('items-center gap-3'):
                                        ui.icon('database', color='secondary').classes('text-purple-600')
                                        db_status = '✅ Exists' if stats.get('vector_db_exists') else '❌ Missing'
                                        ui.label(f"Vector Database: {db_status}").classes('font-semibold text-gray-800')
                            
                            ui.button('Close', on_click=dialog.close, icon='close') \
                                .props('flat color=primary').classes('self-end rounded-lg')
                    
                    dialog.open()
                else:
                    ui.notify("Failed to get knowledge statistics", type='negative', position='top')
        except Exception as e:
            ui.notify(f"Error getting knowledge stats: {e}", type='negative', position='top')

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()

# Create and configure the app
@ui.page('/')
async def main_page():
    """Main page with professional chatbot interface"""
    global chatbot_instance
    
    chatbot_instance = RAGChatbot()
    chatbot_instance.init_ui()

# Add cleanup on app shutdown
@ui.page('/shutdown')
async def shutdown():
    """Cleanup on shutdown"""
    global chatbot_instance
    if chatbot_instance:
        await chatbot_instance.cleanup()



if __name__ in {"__main__", "__mp_main__"}:
    print("🚀 Starting RAG Chatbot UI on http://localhost:8080")
    print("📝 Make sure your FastAPI backend is running on http://localhost:8000")
    print("🎨 Professional UI loaded!")
    
    from starlette.formparsers import MultiPartParser
    MultiPartParser.spool_max_size = 1024 * 1024 * 50  # 50 MB
    
    ui.run(
        title="RAG Document Chatbot",
        port=8080,
        reload=True,  # enable auto-reload
        show_welcome_message=False,
        favicon="🤖",
        uvicorn_logging_level="warning"
    )
