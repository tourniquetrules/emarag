import gradio as gr
import requests
import json
import time
import re
import PyPDF2
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple
import tempfile
from pathlib import Path

# API configuration
LM_STUDIO_BASE_URL = "http://10.5.0.2:1234"
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"
EMBEDDING_MODEL_NAME = "text-embedding-all-minilm-l6-v2-embedding"

# RAG configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_RELEVANT_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.3

# Global variables for chat history and metrics
total_tokens_generated = 0
total_time_spent = 0
total_requests = 0
conversation_id = None
max_conversation_length = 50

# RAG components
embedding_model = None
document_store = []  # List of document chunks with metadata
faiss_index = None
document_embeddings = []

def initialize_embedding_model():
    """Initialize the embedding model (check LM Studio API)"""
    global embedding_model
    try:
        print("🔄 Checking LM Studio embedding API...")
        # Test the embedding API
        test_response = call_embedding_api(["test"])
        if test_response is not None:
            embedding_model = True  # Just use as a flag
            print("✅ LM Studio embedding API is accessible")
            return True
        else:
            print("❌ LM Studio embedding API not accessible")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to embedding API: {e}")
        return False

def call_embedding_api(texts):
    """Call LM Studio embedding API"""
    url = f"{LM_STUDIO_BASE_URL}/v1/embeddings"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "input": texts
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        embeddings = []
        
        # Extract embeddings from response
        for item in data.get("data", []):
            embeddings.append(item["embedding"])
        
        return np.array(embeddings)
        
    except Exception as e:
        print(f"❌ Embedding API Error: {e}")
        return None

def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 4 characters for most models)"""
    return len(text) // 4

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within a reasonable range
            for i in range(min(100, chunk_size // 4)):
                if end - i > start and text[end - i] in '.!?':
                    end = end - i + 1
                    break
        
        chunks.append(text[start:end].strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        # Handle both file path and file object
        if isinstance(pdf_file, str):
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        else:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""

def process_pdf_document(pdf_file, doc_title: str = None) -> int:
    """Process a PDF document and add it to the knowledge base"""
    global document_store, faiss_index, document_embeddings, embedding_model
    
    if embedding_model is None:
        if not initialize_embedding_model():
            return 0
    
    try:
        # Extract text from PDF
        print(f"📄 Processing PDF: {doc_title or 'Uploaded document'}")
        text = extract_text_from_pdf(pdf_file)
        
        if not text:
            print("❌ No text extracted from PDF")
            return 0
        
        # Clean and preprocess text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n+', '\n', text)  # Normalize line breaks
        
        # Chunk the text
        chunks = chunk_text(text)
        print(f"📝 Created {len(chunks)} chunks from document")
        
        # Generate embeddings for chunks
        print(f"🔗 Generating embeddings via LM Studio API...")
        chunk_embeddings = call_embedding_api(chunks)
        
        if chunk_embeddings is None:
            print("❌ Failed to generate embeddings")
            return 0
        
        # Add chunks to document store
        doc_id = len(document_store)
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            document_store.append({
                'id': f"{doc_id}_{i}",
                'text': chunk,
                'source': doc_title or f"Document_{doc_id}",
                'chunk_id': i,
                'embedding': embedding
            })
        
        # Update FAISS index
        if faiss_index is None:
            # Initialize FAISS index
            dimension = chunk_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            document_embeddings = []
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        faiss_index.add(normalized_embeddings.astype('float32'))
        document_embeddings.extend(normalized_embeddings)
        
        print(f"✅ Successfully processed {len(chunks)} chunks")
        return len(chunks)
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return 0

def retrieve_relevant_chunks(query: str, k: int = MAX_RELEVANT_CHUNKS) -> List[Dict]:
    """Retrieve relevant chunks for a query"""
    global embedding_model, faiss_index, document_store
    
    if embedding_model is None or faiss_index is None or not document_store:
        return []
    
    try:
        # Encode query via API
        query_embedding = call_embedding_api([query])
        
        if query_embedding is None:
            print("❌ Failed to encode query")
            return []
            
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search FAISS index
        scores, indices = faiss_index.search(query_embedding.astype('float32'), k)
        
        # Filter by similarity threshold and return relevant chunks
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= SIMILARITY_THRESHOLD and idx < len(document_store):
                chunk = document_store[idx].copy()
                chunk['similarity_score'] = float(score)
                relevant_chunks.append(chunk)
        
        # Sort by similarity score
        relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"🔍 Retrieved {len(relevant_chunks)} relevant chunks (threshold: {SIMILARITY_THRESHOLD})")
        for chunk in relevant_chunks:
            print(f"   📄 {chunk['source']} (similarity: {chunk['similarity_score']:.3f})")
        
        return relevant_chunks
        
    except Exception as e:
        print(f"❌ Error retrieving chunks: {e}")
        return []

def create_rag_prompt(query: str, relevant_chunks: List[Dict]) -> str:
    """Create a RAG prompt with context and query"""
    if not relevant_chunks:
        return f"""You are an assistant helping emergency physicians understand their group meeting discussions and decisions. Answer the following question based on your knowledge of emergency medicine group dynamics and meeting practices:

Question: {query}

Please provide a helpful answer focused on emergency medicine group meeting context, decisions, and discussions."""

    context = "\n\n".join([
        f"Meeting Minutes: {chunk['source']}\nContent: {chunk['text']}"
        for chunk in relevant_chunks
    ])
    
    prompt = f"""You are an assistant helping emergency physicians understand their group meeting discussions and decisions. Use the following meeting minutes to answer the question. Focus on providing accurate information from the meeting records.

CONTEXT FROM MEETING MINUTES:
{context}

QUESTION: {query}

Please provide a comprehensive answer that:
1. References the specific meeting minutes when relevant
2. Focuses on decisions, discussions, and action items from the meetings
3. Highlights any deadlines, assignments, or follow-up items
4. Summarizes relevant policy changes or group decisions
5. Mentions any voting outcomes or consensus reached

If the provided meeting minutes don't fully address the question, clearly state what information is not available in the current meeting records."""

    return prompt

def process_deepseek_response(response_text):
    """Process DeepSeek response to extract the final answer from reasoning"""
    if not response_text:
        return "No response received."
    
    # DeepSeek uses <think> tags for reasoning, extract the final answer
    if "<think>" in response_text and "</think>" in response_text:
        # Split on </think> and get everything after it
        parts = response_text.split("</think>")
        if len(parts) > 1:
            final_answer = parts[-1].strip()
            if final_answer:
                return final_answer
        
        # If no content after </think>, try to extract from inside think tags
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, response_text, re.DOTALL)
        if think_matches:
            # Look for a clear answer in the reasoning
            reasoning = think_matches[-1]
            # Try to find a clear answer pattern
            answer_patterns = [
                r'[Tt]he answer (?:is|should be|would be) (.+?)(?:\.|$)',
                r'[Ss]o (?:the answer is|it\'s) (.+?)(?:\.|$)',
                r'[Tt]herefore,? (.+?)(?:\.|$)',
                r'[Tt]hus,? (.+?)(?:\.|$)',
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, reasoning)
                if match:
                    return match.group(1).strip()
            
            # If no pattern matches, return the last sentence of reasoning
            sentences = reasoning.strip().split('.')
            if sentences and sentences[-1].strip():
                return sentences[-1].strip()
    
    # If no think tags, return the response as-is
    return response_text.strip()

def call_lm_studio_api(messages, temperature=0.7, max_tokens=16384, stream=False):
    """Call LM Studio API with the given messages"""
    url = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    try:
        print(f"🔄 Calling LM Studio API with {len(messages)} messages (streaming: {stream})")
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=120, stream=stream)
        response.raise_for_status()
        
        if stream:
            # Handle streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    full_response += delta['content']
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            raw_response = full_response
            # For streaming, we estimate usage since it's not always provided
            usage = {"completion_tokens": estimate_tokens(raw_response)}
        else:
            # Handle non-streaming response
            end_time = time.time()
            data = response.json()
            raw_response = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
        
        # Process the response to extract the final answer
        processed_response = process_deepseek_response(raw_response)
        
        # Calculate metrics
        response_time = end_time - start_time
        completion_tokens = usage.get("completion_tokens", estimate_tokens(raw_response))
        tokens_per_second = completion_tokens / response_time if response_time > 0 else 0
        
        print(f"✅ API Success: {completion_tokens} tokens in {response_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        print(f"📝 Raw response preview: {raw_response[:150]}...")
        print(f"🎯 Processed answer: {processed_response[:150]}...")
        
        return {
            "success": True,
            "message": processed_response,
            "raw_message": raw_response,
            "response_time": response_time,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tokens_per_second
        }
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error: {e}"
        }

def check_api_health():
    """Check if LM Studio API is available"""
    try:
        response = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_embedding_api_health():
    """Check if LM Studio embedding API is available"""
    try:
        test_embedding = call_embedding_api(["test"])
        return test_embedding is not None
    except:
        return False

def upload_pdf(files):
    """Handle PDF file upload"""
    if not files:
        return "No files uploaded.", ""
    
    results = []
    total_chunks = 0
    
    for file in files:
        file_name = os.path.basename(file.name) if hasattr(file, 'name') else "Unknown file"
        chunks_added = process_pdf_document(file, file_name)
        total_chunks += chunks_added
        
        if chunks_added > 0:
            results.append(f"✅ {file_name}: {chunks_added} chunks processed")
        else:
            results.append(f"❌ {file_name}: Failed to process")
    
    summary = f"📊 **Upload Summary:**\n" + "\n".join(results)
    summary += f"\n\n**Total meeting minutes in knowledge base:** {len(document_store)} chunks"
    
    knowledge_status = f"📚 Knowledge Base: {len(document_store)} chunks from {len(set(chunk['source'] for chunk in document_store))} meeting files"
    
    return summary, knowledge_status

def rag_chat_response(message, history, temperature, max_tokens, system_prompt, show_reasoning, enable_streaming, context_length, enable_rag):
    """Generate RAG-enhanced chat response"""
    global total_tokens_generated, total_time_spent, total_requests, conversation_id, max_conversation_length
    
    # Update context length from UI
    max_conversation_length = int(context_length)
    
    if not message.strip():
        return history, "Please enter a message.", "", ""
    
    # Check for @llm prefix to bypass RAG
    bypass_rag = False
    original_message = message
    if message.lower().startswith('@llm '):
        bypass_rag = True
        message = message[5:].strip()  # Remove @llm prefix
        print("🔄 @llm detected - bypassing RAG for general knowledge query")
    elif message.lower().startswith('@llm'):
        bypass_rag = True
        message = message[4:].strip()  # Remove @llm prefix
        print("🔄 @llm detected - bypassing RAG for general knowledge query")
    
    # Generate conversation ID if this is the first message
    if conversation_id is None:
        conversation_id = f"conv_{int(time.time())}"
        print(f"🆔 Started new conversation: {conversation_id}")
    
    # Check API health
    if not check_api_health():
        return history, f"❌ Cannot connect to LM Studio API at {LM_STUDIO_BASE_URL}", "", ""
    
    # RAG retrieval (skip if @llm prefix used)
    relevant_chunks = []
    sources_info = ""
    
    if bypass_rag:
        sources_info = "🚀 **@llm mode** - Using general knowledge only (RAG bypassed)"
    elif enable_rag and document_store:
        relevant_chunks = retrieve_relevant_chunks(message)
        if relevant_chunks:
            sources = list(set(chunk['source'] for chunk in relevant_chunks))
            sources_info = f"📚 **Meeting minutes consulted:** {', '.join(sources)}\n**Chunks retrieved:** {len(relevant_chunks)}"
        else:
            sources_info = "📚 **No relevant meeting minutes found** - Using general knowledge only"
    else:
        sources_info = "📚 **RAG disabled** - Using general knowledge only"
    
    # Create RAG-enhanced prompt (skip RAG if @llm prefix used)
    if not bypass_rag and enable_rag and relevant_chunks:
        enhanced_message = create_rag_prompt(message, relevant_chunks)
    elif bypass_rag:
        # For @llm mode, use the question directly without meeting context
        enhanced_message = message
    else:
        enhanced_message = f"""You are an assistant helping emergency physicians understand their group meeting discussions and decisions. Answer the following question based on your knowledge of emergency medicine group dynamics and meeting practices:

Question: {message}

Please provide a helpful answer focused on emergency medicine group meeting context, decisions, and discussions."""
    
    # Prepare messages for API with context management
    messages = []
    
    # Add system prompt if provided (but modify for @llm mode)
    if system_prompt.strip():
        if bypass_rag:
            # For @llm mode, use a general assistant prompt
            general_prompt = "You are a helpful, knowledgeable AI assistant. Provide accurate, clear, and concise answers to questions across various topics."
            messages.append({"role": "system", "content": general_prompt})
        else:
            messages.append({"role": "system", "content": system_prompt.strip()})
    
    # Manage conversation length to prevent context overflow
    context_history = history[-max_conversation_length:] if len(history) > max_conversation_length else history
    
    # Add chat history (maintaining full conversation context)
    for user_msg, bot_msg in context_history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    
    # Add current enhanced message
    messages.append({"role": "user", "content": enhanced_message})
    
    print(f"📤 Sending {len(messages)} messages to API (Context: {len(context_history)} pairs)")
    if bypass_rag:
        print(f"🚀 @llm mode: General knowledge query")
    else:
        print(f"🔍 RAG enabled: {enable_rag}, Chunks used: {len(relevant_chunks)}")
    
    # Call API with streaming option
    result = call_lm_studio_api(messages, temperature, max_tokens, stream=enable_streaming)
    
    if result["success"]:
        # Update global metrics
        total_tokens_generated += result["completion_tokens"]
        total_time_spent += result["response_time"]
        total_requests += 1
        
        # Always show the processed (clean) answer in chat history
        chat_display_message = result["message"]
        
        # Add to history - use original user message (with @llm if present)
        history.append([original_message, chat_display_message])
        
        # Generate metrics string
        metrics_str = (f"⏱️ {result['response_time']:.2f}s | "
                      f"🔢 {result['completion_tokens']} tokens | "
                      f"⚡ {result['tokens_per_second']:.1f} tok/s")
        
        # Generate reasoning display
        reasoning_display = ""
        if show_reasoning:
            if result["raw_message"] != result["message"]:
                reasoning_display = f"**Raw Response (with reasoning):**\n```\n{result['raw_message']}\n```\n\n**Processed Answer:**\n{result['message']}"
            else:
                reasoning_display = f"**Response:**\n{result['message']}"
        else:
            reasoning_display = f"**Clean Answer:** {result['message']}"
        
        return history, metrics_str, reasoning_display, sources_info
    else:
        return history, f"❌ Error: {result['message']}", "", sources_info

def clear_chat():
    """Clear chat history and reset conversation state"""
    global total_tokens_generated, total_time_spent, total_requests, conversation_id
    total_tokens_generated = 0
    total_time_spent = 0
    total_requests = 0
    conversation_id = None
    print("🧹 Chat cleared, conversation state reset")
    return [], "Chat cleared and conversation state reset.", "", ""

def clear_knowledge_base():
    """Clear the entire knowledge base"""
    global document_store, faiss_index, document_embeddings
    document_store = []
    faiss_index = None
    document_embeddings = []
    print("🗑️ Knowledge base cleared")
    return "Knowledge base cleared.", "📚 Knowledge Base: Empty"

def load_minutes_directory(minutes_dir="minutes"):
    """Load all PDF files from the minutes directory"""
    global document_store
    
    if not os.path.exists(minutes_dir):
        print(f"📁 Minutes directory '{minutes_dir}' not found")
        return 0, []
    
    pdf_files = list(Path(minutes_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"📁 No PDF files found in '{minutes_dir}' directory")
        return 0, []
    
    print(f"📁 Found {len(pdf_files)} PDF files in '{minutes_dir}' directory")
    
    loaded_files = []
    total_chunks = 0
    
    for pdf_path in pdf_files:
        try:
            file_name = pdf_path.name
            print(f"📄 Loading: {file_name}")
            
            chunks_added = process_pdf_document(str(pdf_path), file_name)
            total_chunks += chunks_added
            
            if chunks_added > 0:
                loaded_files.append(f"✅ {file_name}: {chunks_added} chunks")
                print(f"   ✅ {chunks_added} chunks processed")
            else:
                loaded_files.append(f"❌ {file_name}: Failed to process")
                print(f"   ❌ Failed to process")
                
        except Exception as e:
            loaded_files.append(f"❌ {pdf_path.name}: Error - {str(e)}")
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
    
    print(f"📊 Total: {total_chunks} chunks from {len([f for f in loaded_files if '✅' in f])} successful files")
    return total_chunks, loaded_files

def get_session_stats():
    """Get session statistics"""
    global total_tokens_generated, total_time_spent, total_requests, conversation_id, document_store
    
    if total_requests == 0:
        return "No messages sent yet."
    
    avg_tokens_per_second = total_tokens_generated / total_time_spent if total_time_spent > 0 else 0
    avg_response_time = total_time_spent / total_requests
    
    # Knowledge base stats
    sources = list(set(chunk['source'] for chunk in document_store)) if document_store else []
    
    stats = f"""📊 **Session Statistics:**

**Conversation ID:** {conversation_id or 'No conversation started'}
**Total Requests:** {total_requests}
**Total Tokens Generated:** {total_tokens_generated:,}
**Total Time Spent:** {total_time_spent:.2f} seconds
**Average Response Time:** {avg_response_time:.2f} seconds
**Average Speed:** {avg_tokens_per_second:.1f} tokens/second
**Context Management:** Last {max_conversation_length} message pairs kept

**Meeting Minutes Knowledge Base:**
**Total Chunks:** {len(document_store)}
**Meeting Files:** {len(sources)}
**Sources:** {', '.join(sources) if sources else 'None'}

**Model Configuration:**
**LLM:** {MODEL_NAME}
**Embedding Model:** {EMBEDDING_MODEL_NAME}
**API Endpoint:** {LM_STUDIO_BASE_URL}
**Chunk Size:** {CHUNK_SIZE} tokens
**Max Relevant Chunks:** {MAX_RELEVANT_CHUNKS}
**Similarity Threshold:** {SIMILARITY_THRESHOLD}
"""
    return stats

def reload_minutes():
    """Reload all meeting minutes from the minutes directory"""
    # Clear existing knowledge base
    clear_knowledge_base()
    
    # Load from directory
    total_chunks, loaded_files = load_minutes_directory()
    
    if total_chunks > 0:
        summary = f"📊 **Reload Summary:**\n" + "\n".join(loaded_files)
        summary += f"\n\n**Total chunks loaded:** {total_chunks}"
        knowledge_status = f"📚 Knowledge Base: {len(document_store)} chunks from {len(set(chunk['source'] for chunk in document_store))} meeting files"
    else:
        summary = "❌ No meeting minutes found or failed to load from 'minutes' directory"
        knowledge_status = "📚 Knowledge Base: Empty"
    
    return summary, knowledge_status

# Initialize embedding model on startup
print("🚀 Initializing Emergency Medicine Group Meeting Minutes RAG System...")
embedding_initialized = initialize_embedding_model()

# Auto-load meeting minutes from directory on startup
if embedding_initialized:
    print("📁 Auto-loading meeting minutes from 'minutes' directory...")
    total_chunks, loaded_files = load_minutes_directory()
    if total_chunks > 0:
        print(f"✅ Successfully loaded {total_chunks} chunks from minutes directory")
    else:
        print("ℹ️  No meeting minutes auto-loaded. You can upload PDFs manually or add them to the 'minutes' directory")

# Create the interface
with gr.Blocks(title="Emergency Medicine Group Meeting Minutes RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📋 Emergency Medicine Group Meeting Minutes RAG
    
    Intelligent assistant for searching and understanding your emergency medicine group meeting minutes and decisions.
    
    **LLM:** `deepseek/deepseek-r1-0528-qwen3-8b`
    **Embedding:** `text-embedding-all-minilm-l6-v2-embedding`
    **Focus:** Emergency Medicine Group Meeting Records
    
    💡 **Tip:** Use `@llm` at the start of your question for general knowledge (bypasses meeting minutes search)
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Main chat interface - now at the top
            chatbot = gr.Chatbot(
                label="Meeting Minutes Assistant",
                height=500,
                show_label=True
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Question About Meeting Minutes",
                    placeholder="Ask about decisions, action items, policies, voting outcomes... (Use @llm for general knowledge)",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Metrics and source display
            with gr.Row():
                metrics_display = gr.Textbox(
                    label="Response Metrics",
                    value="Send a message to see metrics...",
                    interactive=False,
                    lines=1,
                    scale=2
                )
                sources_display = gr.Textbox(
                    label="Meeting Minutes Referenced",
                    value="No sources used yet...",
                    interactive=False,
                    lines=1,
                    scale=2
                )
            
            # Reasoning display
            reasoning_display = gr.Markdown(
                label="Response Details",
                value="Response details will appear here...",
            )
            
            # Meeting Minutes Management - moved to bottom
            with gr.Group():
                gr.Markdown("### 📚 Meeting Minutes Management")
                
                with gr.Row():
                    pdf_upload = gr.Files(
                        label="Upload Additional Meeting Minutes (PDF)",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                    upload_btn = gr.Button("Process Minutes", variant="primary")
                
                with gr.Row():
                    reload_btn = gr.Button("Reload from 'minutes' directory", variant="secondary")
                    clear_kb_btn = gr.Button("Clear Knowledge Base", variant="secondary")
                
                upload_status = gr.Textbox(
                    label="Upload/Reload Status",
                    value="Ready to load meeting minutes...",
                    interactive=False,
                    lines=3
                )
                
                knowledge_status = gr.Textbox(
                    label="Knowledge Base Status",
                    value="📚 Loading meeting minutes on startup...",
                    interactive=False,
                    lines=1
                )
            
        with gr.Column(scale=1):
            # Settings panel
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")
                
                enable_rag = gr.Checkbox(
                    label="Enable Meeting Minutes Search",
                    value=True,
                    info="Use uploaded meeting minutes for context"
                )
                
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="You are an assistant helping emergency physicians...",
                    lines=3,
                    value="You are an assistant helping emergency physicians understand their group meeting discussions and decisions. Focus on providing accurate information from meeting records, including decisions made, action items assigned, deadlines set, and policies established."
                )
                
                show_reasoning = gr.Checkbox(
                    label="Show Reasoning Process",
                    value=False,
                    info="Show the full DeepSeek reasoning or just the final answer"
                )
                
                enable_streaming = gr.Checkbox(
                    label="Enable Token Streaming",
                    value=True,
                    info="Stream tokens as they are generated"
                )
                
                context_length = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Conversation Context Length",
                    info="Number of message pairs to keep in context"
                )
                
                gr.Markdown("**Temperature:** Factual ← → Creative")
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.1,
                    step=0.1,
                    label="Temperature"
                )
                
                gr.Markdown("**Max Tokens:** Concise ← → Comprehensive")
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=16384,
                    value=4096,
                    step=128,
                    label="Max Tokens"
                )
            
            # Control buttons
            with gr.Group():
                gr.Markdown("### 🔧 Controls")
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                stats_btn = gr.Button("Show Stats", variant="secondary")
            
            # Stats display
            stats_display = gr.Textbox(
                label="Session Statistics",
                value="No statistics yet...",
                interactive=False,
                lines=12
            )
    
    # Event handlers
    upload_btn.click(
        fn=upload_pdf,
        inputs=[pdf_upload],
        outputs=[upload_status, knowledge_status]
    )
    
    reload_btn.click(
        fn=reload_minutes,
        inputs=[],
        outputs=[upload_status, knowledge_status]
    )
    
    send_btn.click(
        fn=rag_chat_response,
        inputs=[msg_input, chatbot, temperature, max_tokens, system_prompt, show_reasoning, enable_streaming, context_length, enable_rag],
        outputs=[chatbot, metrics_display, reasoning_display, sources_display]
    ).then(
        lambda: "",  # Clear the input
        outputs=msg_input
    )
    
    msg_input.submit(
        fn=rag_chat_response,
        inputs=[msg_input, chatbot, temperature, max_tokens, system_prompt, show_reasoning, enable_streaming, context_length, enable_rag],
        outputs=[chatbot, metrics_display, reasoning_display, sources_display]
    ).then(
        lambda: "",  # Clear the input
        outputs=msg_input
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, metrics_display, reasoning_display, sources_display]
    )
    
    clear_kb_btn.click(
        fn=clear_knowledge_base,
        outputs=[upload_status, knowledge_status]
    )
    
    stats_btn.click(
        fn=get_session_stats,
        outputs=stats_display
    )

if __name__ == "__main__":
    print("🚀 Starting Emergency Medicine Group Meeting Minutes RAG Interface...")
    print(f"📡 API Endpoint: {LM_STUDIO_BASE_URL}")
    print(f"🤖 LLM Model: {MODEL_NAME}")
    print(f"🔍 Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"📄 Chunk Size: {CHUNK_SIZE}")
    print(f"🎯 Max Relevant Chunks: {MAX_RELEVANT_CHUNKS}")
    print(f"📊 Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"📁 Minutes Directory: ./minutes")
    
    # Show current knowledge base status
    if document_store:
        sources = list(set(chunk['source'] for chunk in document_store))
        print(f"📚 Knowledge Base: {len(document_store)} chunks from {len(sources)} meeting files")
    else:
        print("📚 Knowledge Base: Empty (no meeting minutes loaded)")
    
    # Check API health at startup
    print("🔍 Checking API connections...")
    if check_api_health():
        print("✅ LM Studio LLM API is accessible")
    else:
        print("❌ Warning: Cannot connect to LM Studio LLM API")
    
    if check_embedding_api_health():
        print("✅ LM Studio Embedding API is accessible")
    else:
        print("❌ Warning: Cannot connect to LM Studio Embedding API")
    
    print("\n" + "="*50)
    print("🌐 Starting web interface...")
    print("📱 Access at: http://localhost:7870")
    print("🔄 To reload meeting minutes, use the 'Reload from minutes directory' button")
    print("=" * 50)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7870,  # Port 7870 as requested
        share=False,
        show_error=True
    )
