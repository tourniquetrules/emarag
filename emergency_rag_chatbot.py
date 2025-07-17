import gradio as gr
import requests
import json
import time
import re
import pypdf
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Python version compatibility check
if sys.version_info < (3, 9):
    print("❌ Error: This application requires Python 3.9 or higher")
    print(f"   Current version: {sys.version}")
    sys.exit(1)
elif sys.version_info >= (3, 13):
    print(f"⚠️  Warning: Python {sys.version_info.major}.{sys.version_info.minor} is newer than tested versions")
    print("   This application was optimized for Python 3.9-3.12")
else:
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} compatibility verified")

# Medical embedding imports
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    import torch
    MEDICAL_EMBEDDINGS_AVAILABLE = True
    print("✅ Medical embedding libraries available")
except ImportError as e:
    MEDICAL_EMBEDDINGS_AVAILABLE = False
    print(f"⚠️  Medical embedding libraries not available: {e}")
    print("📦 Run: pip install sentence-transformers transformers torch")

# OpenAI imports
try:
    import openai
    OPENAI_AVAILABLE = True
    print("✅ OpenAI library available")
except ImportError as e:
    OPENAI_AVAILABLE = False
    print(f"⚠️  OpenAI library not available: {e}")
    print("📦 Run: pip install openai python-dotenv")

# Enhanced RAG imports
try:
    import spacy
    SPACY_AVAILABLE = True
    print("✅ spaCy available for enhanced chunking")
except ImportError as e:
    SPACY_AVAILABLE = False
    print(f"⚠️  spaCy not available: {e}")
    print("📦 Run: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
    print("✅ Cross-encoder available for reranking")
except ImportError as e:
    CROSS_ENCODER_AVAILABLE = False
    print(f"⚠️  Cross-encoder not available: {e}")
    print("📦 Already have sentence-transformers, cross-encoder should work")

# Global configuration
USE_OPENAI = False  # Will be set based on user choice
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API configuration - LM Studio (Local AI)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://192.168.2.64:1234")  # Default fallback
LM_STUDIO_MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"
LM_STUDIO_EMBEDDING_MODEL_NAME = "text-embedding-all-minilm-l6-v2-embedding"  # Fallback for LM Studio

# OpenAI configuration
OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-large"

# Dynamic model names (will be set based on provider choice)
MODEL_NAME = LM_STUDIO_MODEL_NAME
EMBEDDING_MODEL_NAME = LM_STUDIO_EMBEDDING_MODEL_NAME

# Clinical-BERT model for medical embeddings
# Note: This model is cached after first download (~1GB) in ~/.cache/huggingface/
# The "Creating new one with mean pooling" message appears each startup but is just
# a fast conversion step, not a re-download. Alternative pre-optimized models:
# - "sentence-transformers/all-MiniLM-L6-v2" (general, faster startup)
# - "sentence-transformers/all-mpnet-base-v2" (better quality, general)
# - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" (medical)
CLINICAL_BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"  # Primary medical embedding model

# RAG configuration - Enhanced for medical content
CHUNK_SIZE = 1024  # Increased from 512 for better medical context
CHUNK_OVERLAP = 100  # Increased from 50 for better continuity
MAX_RELEVANT_CHUNKS = 10  # Increased from 5 for more comprehensive answers
SIMILARITY_THRESHOLD = 0.25  # Decreased from 0.3 for better recall

# Global variables for chat history and metrics
total_tokens_generated = 0
total_time_spent = 0
total_requests = 0
conversation_id = None
max_conversation_length = 50

# RAG components
embedding_model = None
clinical_bert_model = None  # Clinical-BERT model instance
document_store = []  # List of document chunks with metadata
faiss_index = None
document_embeddings = []
use_clinical_bert = True  # Primary embedding method

# Enhanced RAG components
spacy_nlp = None  # spaCy model for enhanced chunking
cross_encoder = None  # Cross-encoder for reranking

def choose_ai_provider():
    """Ask user to choose between Local AI (LM Studio) or OpenAI"""
    global USE_OPENAI, MODEL_NAME, EMBEDDING_MODEL_NAME
    
    print("\n" + "="*60)
    print("🚀 Emergency Medicine RAG Chat Interface Setup")
    print("="*60)
    print("Choose your AI provider:")
    print("1. 🏠 Local AI (LM Studio) - Private, runs on your hardware")
    print("2. ☁️  OpenAI - Cloud-based, requires API key")
    print("="*60)
    
    while True:
        choice = input("Enter your choice (1 for Local AI, 2 for OpenAI): ").strip()
        
        if choice == "1":
            USE_OPENAI = False
            MODEL_NAME = LM_STUDIO_MODEL_NAME
            EMBEDDING_MODEL_NAME = LM_STUDIO_EMBEDDING_MODEL_NAME
            print("✅ Using Local AI (LM Studio)")
            print(f"📡 LLM Model: {MODEL_NAME}")
            print(f"🔍 Embedding Model: {EMBEDDING_MODEL_NAME}")
            break
        elif choice == "2":
            if not OPENAI_AVAILABLE:
                print("❌ OpenAI library not available. Please install: pip install openai python-dotenv")
                continue
            if not OPENAI_API_KEY:
                print("❌ OpenAI API key not found in .env file")
                print("📝 Please create a .env file with: OPENAI_API_KEY=your_api_key_here")
                continue
            
            USE_OPENAI = True
            MODEL_NAME = OPENAI_MODEL_NAME
            EMBEDDING_MODEL_NAME = OPENAI_EMBEDDING_MODEL_NAME
            # Initialize OpenAI client
            openai.api_key = OPENAI_API_KEY
            print("✅ Using OpenAI")
            print(f"📡 LLM Model: {MODEL_NAME}")
            print(f"🔍 Embedding Model: {EMBEDDING_MODEL_NAME}")
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
    
    print("="*60 + "\n")

def initialize_embedding_model():
    """Initialize the embedding model (Clinical-BERT preferred, LM Studio fallback)"""
    global embedding_model, clinical_bert_model, use_clinical_bert, spacy_nlp, cross_encoder
    
    # Initialize spaCy for enhanced chunking
    if SPACY_AVAILABLE:
        try:
            # Try medical spaCy model first
            try:
                spacy_nlp = spacy.load("en_core_sci_md")
                print("✅ Loaded scientific spaCy model (en_core_sci_md)")
            except OSError:
                # Fallback to standard model
                spacy_nlp = spacy.load("en_core_web_sm")
                print("✅ Loaded standard spaCy model (en_core_web_sm)")
                print("💡 Consider installing scientific model: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz")
        except Exception as e:
            print(f"⚠️  Failed to load spaCy model: {e}")
            spacy_nlp = None
    
    # Initialize cross-encoder for reranking
    if CROSS_ENCODER_AVAILABLE:
        try:
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Loaded cross-encoder for reranking")
        except Exception as e:
            print(f"⚠️  Failed to load cross-encoder: {e}")
            cross_encoder = None
    
    # Try to initialize Clinical-BERT first
    if MEDICAL_EMBEDDINGS_AVAILABLE:
        try:
            print("🏥 Initializing Clinical-BERT for medical embeddings...")
            clinical_bert_model = SentenceTransformer(CLINICAL_BERT_MODEL)
            use_clinical_bert = True
            embedding_model = True
            print("✅ Clinical-BERT loaded successfully")
            print(f"📏 Embedding dimension: {clinical_bert_model.get_sentence_embedding_dimension()}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load Clinical-BERT: {e}")
            print("🔄 Falling back to LM Studio API...")
    
    # Fallback to LM Studio API
    try:
        print("🔄 Checking LM Studio embedding API...")
        test_response = call_embedding_api(["test"])
        if test_response is not None:
            embedding_model = True
            use_clinical_bert = False
            print("✅ LM Studio embedding API is accessible (fallback mode)")
            return True
        else:
            print("❌ LM Studio embedding API not accessible")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to embedding API: {e}")
        return False

def get_clinical_bert_embeddings(texts):
    """Generate embeddings using Clinical-BERT"""
    global clinical_bert_model
    
    if clinical_bert_model is None:
        return None
    
    try:
        print(f"🏥 Generating Clinical-BERT embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        # Generate embeddings
        embeddings = clinical_bert_model.encode(
            texts,
            batch_size=8,  # Reasonable batch size
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for cosine similarity
        )
        
        end_time = time.time()
        print(f"✅ Clinical-BERT embeddings generated in {end_time - start_time:.2f}s")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Clinical-BERT embedding error: {e}")
        return None

def get_openai_embeddings(texts):
    """Generate embeddings using OpenAI API"""
    if not USE_OPENAI or not OPENAI_API_KEY:
        return None
    
    try:
        print(f"☁️  Generating OpenAI embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        # OpenAI API call
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=texts
        )
        
        # Extract embeddings
        embeddings = np.array([item.embedding for item in response.data])
        
        end_time = time.time()
        print(f"✅ OpenAI embeddings generated in {end_time - start_time:.2f}s")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ OpenAI embedding error: {e}")
        return None

def call_embedding_api(texts):
    """Generate embeddings using Clinical-BERT (preferred), OpenAI, or LM Studio API (fallback)"""
    global use_clinical_bert
    
    # Try Clinical-BERT first (if available and not using OpenAI exclusively)
    if use_clinical_bert and clinical_bert_model is not None and not USE_OPENAI:
        embeddings = get_clinical_bert_embeddings(texts)
        if embeddings is not None:
            return embeddings
        else:
            print("⚠️  Clinical-BERT failed, falling back to LM Studio...")
            use_clinical_bert = False
    
    # Use OpenAI if selected
    if USE_OPENAI:
        embeddings = get_openai_embeddings(texts)
        if embeddings is not None:
            return embeddings
        else:
            print("⚠️  OpenAI embeddings failed, falling back to LM Studio...")
    
    # Fallback to LM Studio API
    print("🔄 Using LM Studio embeddings as fallback...")
    return get_lm_studio_embeddings(texts)

def get_lm_studio_embeddings(texts):
    """Generate embeddings using LM Studio API"""
    url = f"{LM_STUDIO_BASE_URL}/v1/embeddings"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LM_STUDIO_EMBEDDING_MODEL_NAME,
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
        print(f"❌ LM Studio Embedding API Error: {e}")
        return None

def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 4 characters for most models)"""
    return len(text) // 4

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Enhanced text chunking with sentence-aware segmentation"""
    global spacy_nlp
    
    # Try spaCy enhanced chunking first
    if SPACY_AVAILABLE and spacy_nlp is not None:
        try:
            doc = spacy_nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            chunks = []
            current_chunk = ""
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence would exceed chunk size, save current chunk
                if current_length + sentence_length > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Create overlap from end of current chunk
                    words = current_chunk.split()
                    overlap_words = words[-overlap//4:] if len(words) > overlap//4 else words
                    current_chunk = " ".join(overlap_words) + " " + sentence
                    current_length = len(current_chunk)
                else:
                    # Add sentence to current chunk
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_length += sentence_length
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            print(f"✅ Enhanced spaCy chunking: {len(chunks)} chunks from {len(sentences)} sentences")
            return chunks
            
        except Exception as e:
            print(f"⚠️  spaCy chunking failed, using fallback: {e}")
    
    # Original chunking method (fallback)
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
    
    print(f"📝 Standard chunking: {len(chunks)} chunks")
    return chunks

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        # Handle both file path and file object
        if isinstance(pdf_file, str):
            with open(pdf_file, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        else:
            pdf_reader = pypdf.PdfReader(pdf_file)
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
        if USE_OPENAI:
            print(f"🔗 Generating embeddings via OpenAI API...")
        else:
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
            # Initialize FAISS index with correct dimension
            dimension = chunk_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            document_embeddings = []
            print(f"🗂️  Initialized FAISS index with dimension {dimension}")
        
        # Normalize embeddings for cosine similarity (Clinical-BERT already normalized)
        if use_clinical_bert:
            normalized_embeddings = chunk_embeddings  # Already normalized
        else:
            normalized_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        faiss_index.add(normalized_embeddings.astype('float32'))
        document_embeddings.extend(normalized_embeddings)
        
        print(f"✅ Successfully processed {len(chunks)} chunks")
        return len(chunks)
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return 0

def rerank_chunks_with_cross_encoder(query: str, chunks: List[Dict]) -> List[Dict]:
    """Rerank retrieved chunks using cross-encoder for better relevance"""
    global cross_encoder
    
    if not CROSS_ENCODER_AVAILABLE or cross_encoder is None or len(chunks) <= 1:
        return chunks
    
    try:
        print(f"🔄 Reranking {len(chunks)} chunks with cross-encoder...")
        
        # Prepare query-chunk pairs for cross-encoder
        pairs = [[query, chunk['text']] for chunk in chunks]
        
        # Get cross-encoder scores
        scores = cross_encoder.predict(pairs)
        
        # Add scores to chunks and sort
        for chunk, score in zip(chunks, scores):
            chunk['cross_encoder_score'] = float(score)
            chunk['original_similarity'] = chunk.get('similarity_score', 0.0)
        
        # Sort by cross-encoder score (descending)
        reranked_chunks = sorted(chunks, key=lambda x: x['cross_encoder_score'], reverse=True)
        
        print(f"✅ Reranked chunks by medical relevance")
        for i, chunk in enumerate(reranked_chunks[:3]):  # Show top 3
            print(f"   {i+1}. Cross-encoder: {chunk['cross_encoder_score']:.3f}, Original: {chunk['original_similarity']:.3f}")
            print(f"      {chunk['text'][:100]}...")
        
        return reranked_chunks
        
    except Exception as e:
        print(f"⚠️  Cross-encoder reranking failed: {e}")
        return chunks

def retrieve_relevant_chunks(query: str, k: int = MAX_RELEVANT_CHUNKS) -> List[Dict]:
    """Retrieve relevant chunks for a query with enhanced reranking"""
    global embedding_model, faiss_index, document_store
    
    if embedding_model is None or faiss_index is None or not document_store:
        return []
    
    try:
        # Encode query via API
        query_embedding = call_embedding_api([query])
        
        if query_embedding is None:
            print("❌ Failed to encode query")
            return []
            
        # Normalize query embedding (Clinical-BERT already normalized)
        if use_clinical_bert:
            query_embedding_norm = query_embedding  # Already normalized
        else:
            query_embedding_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search FAISS index - get more chunks for reranking
        search_k = min(k * 2, len(document_store))  # Get 2x chunks for reranking
        scores, indices = faiss_index.search(query_embedding_norm.astype('float32'), search_k)
        
        # Filter by similarity threshold and return relevant chunks
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= SIMILARITY_THRESHOLD and idx < len(document_store):
                chunk = document_store[idx].copy()
                chunk['similarity_score'] = float(score)
                relevant_chunks.append(chunk)
        
        # Sort by similarity score initially
        relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"🔍 Retrieved {len(relevant_chunks)} chunks from FAISS (threshold: {SIMILARITY_THRESHOLD})")
        
        # Apply cross-encoder reranking if available
        if len(relevant_chunks) > 1:
            relevant_chunks = rerank_chunks_with_cross_encoder(query, relevant_chunks)
        
        # Return top k chunks after reranking
        final_chunks = relevant_chunks[:k]
        
        print(f"📋 Final selection: {len(final_chunks)} chunks after reranking")
        for i, chunk in enumerate(final_chunks):
            ce_score = chunk.get('cross_encoder_score', 'N/A')
            print(f"   {i+1}. 📄 {chunk['source']} (similarity: {chunk['similarity_score']:.3f}, relevance: {ce_score})")
        
        return final_chunks
        
    except Exception as e:
        print(f"❌ Error retrieving chunks: {e}")
        return []

def create_rag_prompt(query: str, relevant_chunks: List[Dict]) -> str:
    """Create a RAG prompt with context and query"""
    if not relevant_chunks:
        return f"""You are an emergency medicine expert. Answer the following question based on your knowledge:

Question: {query}

Please provide a comprehensive answer focused on emergency medicine practices, protocols, and evidence-based care."""

    context = "\n\n".join([
        f"Source: {chunk['source']}\nContent: {chunk['text']}"
        for chunk in relevant_chunks
    ])
    
    prompt = f"""You are an emergency medicine expert. Use the following research abstracts and medical literature to answer the question. Focus on evidence-based practices, clinical protocols, and emergency medicine guidelines.

CONTEXT FROM MEDICAL LITERATURE:
{context}

QUESTION: {query}

Please provide a comprehensive answer that:
1. References the provided medical literature when relevant
2. Focuses on emergency medicine practices and protocols
3. Includes evidence-based recommendations
4. Considers safety and time-sensitive aspects of emergency care

If the provided context doesn't fully address the question, supplement with your emergency medicine knowledge while clearly distinguishing between context-based and general knowledge."""

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

def call_openai_api(messages, temperature=0.7, max_tokens=16384, stream=False):
    """Call OpenAI API with the given messages"""
    if not USE_OPENAI or not OPENAI_API_KEY:
        return {"success": False, "error": "OpenAI not configured", "message": "OpenAI not configured"}
    
    try:
        print(f"☁️  Calling OpenAI API with {len(messages)} messages (streaming: {stream})")
        
        start_time = time.time()
        
        # Note: Some OpenAI models (o3, o4-mini) only support default temperature (1.0)
        # Most models like gpt-4o-mini support custom temperature values
        api_params = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_completion_tokens": max_tokens,  # Use max_completion_tokens for newer models
            "stream": stream
        }
        
        # Only add temperature if it's not the default value for specific models
        if MODEL_NAME in ["o4-mini-2025-04-16", "o3-2025-04-16"]:
            # These models only support default temperature
            print(f"ℹ️  Using default temperature (1.0) for {MODEL_NAME}")
        else:
            # Most models support custom temperature
            api_params["temperature"] = temperature
        
        response = openai.chat.completions.create(**api_params)
        
        if stream:
            # Handle streaming response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
            
            end_time = time.time()
            raw_response = full_response
            completion_tokens = estimate_tokens(raw_response)
        else:
            # Handle non-streaming response
            end_time = time.time()
            raw_response = response.choices[0].message.content
            completion_tokens = response.usage.completion_tokens if response.usage else estimate_tokens(raw_response)
        
        # Process the response (OpenAI doesn't use <think> tags like DeepSeek)
        processed_response = raw_response.strip()
        
        # Calculate metrics
        response_time = end_time - start_time
        tokens_per_second = completion_tokens / response_time if response_time > 0 else 0
        
        print(f"✅ OpenAI API Success: {completion_tokens} tokens in {response_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        print(f"📝 Response preview: {processed_response[:150]}...")
        
        return {
            "success": True,
            "message": processed_response,
            "raw_message": raw_response,
            "response_time": response_time,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tokens_per_second
        }
        
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"OpenAI Error: {e}"
        }

def call_lm_studio_api(messages, temperature=0.7, max_tokens=16384, stream=False):
    """Call LM Studio API with the given messages"""
    url = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add a conversation reset to ensure clean state
    payload = {
        "model": LM_STUDIO_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "stop": ["<|endoftext|>", "<|end|>"],  # Add stop tokens to prevent contamination
        "presence_penalty": 0.1  # Slight penalty to prevent repetition
    }
    
    try:
        print(f"🏠 Calling LM Studio API with {len(messages)} messages (streaming: {stream})")
        
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
        
        # Process the response to extract the final answer (DeepSeek specific)
        processed_response = process_deepseek_response(raw_response)
        
        # Calculate metrics
        response_time = end_time - start_time
        completion_tokens = usage.get("completion_tokens", estimate_tokens(raw_response))
        tokens_per_second = completion_tokens / response_time if response_time > 0 else 0
        
        print(f"✅ LM Studio API Success: {completion_tokens} tokens in {response_time:.2f}s ({tokens_per_second:.1f} tok/s)")
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
        print(f"❌ LM Studio API Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"LM Studio Error: {e}"
        }

def call_ai_api(messages, temperature=0.7, max_tokens=16384, stream=False):
    """Call the appropriate AI API based on user choice"""
    if USE_OPENAI:
        return call_openai_api(messages, temperature, max_tokens, stream)
    else:
        return call_lm_studio_api(messages, temperature, max_tokens, stream)

def check_api_health():
    """Check if the selected AI API is available"""
    if USE_OPENAI:
        try:
            # Test OpenAI API with a simple completion
            test_response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "Hi"}],
                max_completion_tokens=10  # Increase to allow for a simple response
            )
            return True
        except Exception as e:
            print(f"❌ OpenAI API health check failed: {e}")
            return False
    else:
        try:
            response = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ LM Studio API health check failed: {e}")
            return False

def check_embedding_api_health():
    """Check if the selected embedding API is available"""
    try:
        test_embedding = call_embedding_api(["test"])
        return test_embedding is not None
    except:
        return False

def upload_pdf(files):
    """Handle PDF file upload and save to abstracts directory"""
    if not files:
        return "No files uploaded.", ""
    
    # Ensure abstracts directory exists
    abstracts_dir = "abstracts"
    os.makedirs(abstracts_dir, exist_ok=True)
    
    results = []
    total_chunks = 0
    saved_files = []
    
    for file in files:
        file_name = os.path.basename(file.name) if hasattr(file, 'name') else "Unknown file"
        
        # Process the PDF for RAG
        chunks_added = process_pdf_document(file, file_name)
        total_chunks += chunks_added
        
        # Save the PDF to abstracts directory permanently
        try:
            if hasattr(file, 'name') and file.name:
                # Copy the uploaded file to abstracts directory
                abstracts_path = os.path.join(abstracts_dir, file_name)
                
                # Avoid overwriting existing files
                counter = 1
                original_name = file_name
                while os.path.exists(abstracts_path):
                    name_parts = original_name.rsplit('.', 1)
                    if len(name_parts) == 2:
                        file_name = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    else:
                        file_name = f"{original_name}_{counter}"
                    abstracts_path = os.path.join(abstracts_dir, file_name)
                    counter += 1
                
                # Copy file to permanent location
                import shutil
                shutil.copy2(file.name, abstracts_path)
                saved_files.append(file_name)
                
                if chunks_added > 0:
                    results.append(f"✅ {file_name}: {chunks_added} chunks processed & saved")
                else:
                    results.append(f"⚠️  {file_name}: Saved but processing failed")
            else:
                if chunks_added > 0:
                    results.append(f"✅ {file_name}: {chunks_added} chunks processed (not saved - no file path)")
                else:
                    results.append(f"❌ {file_name}: Failed to process")
                    
        except Exception as e:
            if chunks_added > 0:
                results.append(f"✅ {file_name}: {chunks_added} chunks processed (save failed: {str(e)})")
            else:
                results.append(f"❌ {file_name}: Processing and save failed")
    
    summary = f"📊 **Upload Summary:**\n" + "\n".join(results)
    summary += f"\n\n**Total documents in knowledge base:** {len(document_store)} chunks"
    
    if saved_files:
        summary += f"\n**📁 Saved to abstracts/:** {', '.join(saved_files)}"
    
    knowledge_status = f"📚 Knowledge Base: {len(document_store)} chunks from {len(set(chunk['source'] for chunk in document_store))} documents"
    
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
    api_name = "OpenAI" if USE_OPENAI else "LM Studio"
    if not check_api_health():
        return history, f"❌ Cannot connect to {api_name} API", "", ""
    
    # RAG retrieval (skip if @llm prefix used)
    relevant_chunks = []
    sources_info = ""
    
    if bypass_rag:
        sources_info = "🚀 **@llm mode** - Using general knowledge only (RAG bypassed)"
    elif enable_rag and document_store:
        relevant_chunks = retrieve_relevant_chunks(message)
        if relevant_chunks:
            sources = list(set(chunk['source'] for chunk in relevant_chunks))
            sources_info = f"📚 **Sources consulted:** {', '.join(sources)}\n**Chunks retrieved:** {len(relevant_chunks)}"
        else:
            sources_info = "📚 **No relevant chunks found** - Using general knowledge only"
    else:
        sources_info = "📚 **RAG disabled** - Using general knowledge only"
    
    # Create RAG-enhanced prompt (skip RAG if @llm prefix used)
    if not bypass_rag and enable_rag and relevant_chunks:
        enhanced_message = create_rag_prompt(message, relevant_chunks)
    elif bypass_rag:
        # For @llm mode, use the question directly without emergency medicine context
        enhanced_message = message
    else:
        enhanced_message = f"""You are an emergency medicine expert. Answer the following question based on your knowledge:

Question: {message}

Please provide a comprehensive answer focused on emergency medicine practices, protocols, and evidence-based care."""
    
    # Prepare messages for API with context management
    messages = []
    
    # Add system prompt if provided (but modify for @llm mode)
    if system_prompt.strip():
        if bypass_rag:
            # For @llm mode, use a general assistant prompt instead of emergency medicine
            general_prompt = "You are a helpful, knowledgeable AI assistant. Provide accurate, clear, and concise answers to questions across various topics."
            messages.append({"role": "system", "content": general_prompt})
        else:
            messages.append({"role": "system", "content": system_prompt.strip()})
    
    # Manage conversation length to prevent context overflow
    context_history = history[-(max_conversation_length*2):] if len(history) > max_conversation_length*2 else history
    
    # Add chat history (maintaining full conversation context)
    for msg in context_history:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current enhanced message
    messages.append({"role": "user", "content": enhanced_message})
    
    print(f"📤 Sending {len(messages)} messages to API (Context: {len(context_history)//2} pairs)")
    if bypass_rag:
        print(f"🚀 @llm mode: General knowledge query")
    else:
        print(f"🔍 RAG enabled: {enable_rag}, Chunks used: {len(relevant_chunks)}")
    
    # Call API with streaming option
    result = call_ai_api(messages, temperature, max_tokens, stream=enable_streaming)
    
    if result["success"]:
        # Update global metrics
        total_tokens_generated += result["completion_tokens"]
        total_time_spent += result["response_time"]
        total_requests += 1
        
        # Always show the processed (clean) answer in chat history
        chat_display_message = result["message"]
        
        # Add to history - use original user message (with @llm if present)
        history.append({"role": "user", "content": original_message})
        history.append({"role": "assistant", "content": chat_display_message})
        
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

def load_abstracts_directory(abstracts_dir="abstracts"):
    """Load all PDF files from the abstracts directory"""
    global document_store
    
    if not os.path.exists(abstracts_dir):
        print(f"📁 Abstracts directory '{abstracts_dir}' not found")
        return 0, []
    
    pdf_files = list(Path(abstracts_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"📁 No PDF files found in '{abstracts_dir}' directory")
        return 0, []
    
    print(f"📁 Found {len(pdf_files)} PDF files in '{abstracts_dir}' directory")
    
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

def get_embedding_status():
    """Get current embedding model status"""
    if use_clinical_bert and clinical_bert_model is not None and not USE_OPENAI:
        return f"🏥 Clinical-BERT ({CLINICAL_BERT_MODEL})"
    elif USE_OPENAI:
        return f"☁️  OpenAI ({EMBEDDING_MODEL_NAME})"
    elif embedding_model:
        return f"🏠 LM Studio API ({LM_STUDIO_EMBEDDING_MODEL_NAME})"
    else:
        return "❌ No embedding model available"
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

**Knowledge Base:**
**Total Chunks:** {len(document_store)}
**Documents:** {len(sources)}
**Sources:** {', '.join(sources) if sources else 'None'}

**Model Configuration:**
**AI Provider:** {'☁️  OpenAI' if USE_OPENAI else '🏠 Local AI (LM Studio)'}
**LLM:** {MODEL_NAME}
**Embedding Model:** {get_embedding_status()}
**API Endpoint:** {'OpenAI API' if USE_OPENAI else LM_STUDIO_BASE_URL}
**Chunk Size:** {CHUNK_SIZE} tokens
**Max Relevant Chunks:** {MAX_RELEVANT_CHUNKS}
**Similarity Threshold:** {SIMILARITY_THRESHOLD}
"""
    return stats

def reload_abstracts():
    """Reload all abstracts from the abstracts directory"""
    # Clear existing knowledge base
    clear_knowledge_base()
    
    # Load from directory
    total_chunks, loaded_files = load_abstracts_directory()
    
    if total_chunks > 0:
        summary = f"📊 **Reload Summary:**\n" + "\n".join(loaded_files)
        summary += f"\n\n**Total chunks loaded:** {total_chunks}"
        knowledge_status = f"📚 Knowledge Base: {len(document_store)} chunks from {len(set(chunk['source'] for chunk in document_store))} documents"
    else:
        summary = "❌ No abstracts found or failed to load from 'abstracts' directory"
        knowledge_status = "📚 Knowledge Base: Empty"
    
    return summary, knowledge_status

def get_available_openai_models():
    """Fetch all available OpenAI models"""
    if not USE_OPENAI or not OPENAI_API_KEY:
        return {"success": False, "error": "OpenAI not configured"}
    
    try:
        print("🔍 Fetching available OpenAI models...")
        models = openai.models.list()
        
        # Categorize models
        chat_models = []
        embedding_models = []
        other_models = []
        
        for model in sorted(models.data, key=lambda x: x.id):
            model_id = model.id
            if any(prefix in model_id.lower() for prefix in ['gpt', 'o1', 'o3', 'chatgpt']):
                chat_models.append(model_id)
            elif 'embedding' in model_id.lower():
                embedding_models.append(model_id)
            else:
                other_models.append(model_id)
        
        return {
            "success": True,
            "chat_models": chat_models,
            "embedding_models": embedding_models,
            "other_models": other_models,
            "total_count": len(models.data)
        }
        
    except Exception as e:
        print(f"❌ Error fetching OpenAI models: {e}")
        return {"success": False, "error": str(e)}

def display_available_models():
    """Display available OpenAI models in a formatted way"""
    result = get_available_openai_models()
    
    if not result["success"]:
        return f"❌ Failed to fetch models: {result['error']}"
    
    output = ["🤖 **Available OpenAI Models:**\n"]
    
    if result["chat_models"]:
        output.append("**📡 CHAT/COMPLETION MODELS:**")
        for model in result["chat_models"]:
            output.append(f"  • `{model}`")
        output.append("")
    
    if result["embedding_models"]:
        output.append("**🔍 EMBEDDING MODELS:**")
        for model in result["embedding_models"]:
            output.append(f"  • `{model}`")
        output.append("")
    
    if result["other_models"]:
        output.append("**🔧 OTHER MODELS:**")
        for model in result["other_models"]:
            output.append(f"  • `{model}`")
        output.append("")
    
    output.append(f"**📊 SUMMARY:**")
    output.append(f"  • Total Models: {result['total_count']}")
    output.append(f"  • Chat Models: {len(result['chat_models'])}")
    output.append(f"  • Embedding Models: {len(result['embedding_models'])}")
    output.append(f"  • Other Models: {len(result['other_models'])}")
    
    return "\n".join(output)

# Choose AI provider before initialization
choose_ai_provider()

# Initialize embedding model on startup
print("🚀 Initializing Emergency Medicine RAG System...")
embedding_initialized = initialize_embedding_model()

# Auto-load abstracts from directory on startup
if embedding_initialized:
    print("📁 Auto-loading abstracts from 'abstracts' directory...")
    total_chunks, loaded_files = load_abstracts_directory()
    if total_chunks > 0:
        print(f"✅ Successfully loaded {total_chunks} chunks from abstracts directory")
    else:
        print("ℹ️  No abstracts auto-loaded. You can upload PDFs manually or add them to the 'abstracts' directory")

# Create the interface
with gr.Blocks(title="Emergency Medicine RAG Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🚑 Emergency Medicine RAG Chat Interface - Enhanced
    
    Advanced RAG system for emergency medicine using medical literature and abstracts.
    
    **AI Provider:** {'☁️  OpenAI' if USE_OPENAI else '🏠 Local AI (LM Studio)'}
    **LLM:** `{MODEL_NAME}`
    **Embedding:** `Clinical-BERT` (medical-specialized) + `{'OpenAI' if USE_OPENAI else 'LM Studio'}` (fallback)
    **Enhanced Features:** 
    - 🧠 spaCy sentence-aware chunking: {'✅ Available' if SPACY_AVAILABLE else '❌ Not installed'}
    - 🎯 Cross-encoder reranking: {'✅ Available' if CROSS_ENCODER_AVAILABLE else '❌ Not installed'}
    - 📏 Larger chunks (1024 tokens) for better medical context
    - 📚 More chunks retrieved (10) for comprehensive answers
    
    💡 **Tip:** Use `@llm` at the start of your question for general knowledge (bypasses RAG)
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Main chat interface - now at the top
            chatbot = gr.Chatbot(
                label="Emergency Medicine Chat",
                height=500,
                show_label=True,
                type="messages"  # Use modern message format
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Medical Question",
                    placeholder="Ask about emergency medicine protocols, treatments, research... (Use @llm for general knowledge)",
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
                    label="Sources Used",
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
            
            # Knowledge Base Management - moved to bottom
            with gr.Group():
                gr.Markdown("### 📚 Knowledge Base Management")
                
                with gr.Row():
                    pdf_upload = gr.Files(
                        label="Upload Additional PDFs (Auto-saved to abstracts/)",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                    upload_btn = gr.Button("Process PDFs", variant="primary")
                
                with gr.Row():
                    reload_btn = gr.Button("Reload from 'abstracts' directory", variant="secondary")
                    save_temp_btn = gr.Button("Save Uploaded PDFs", variant="secondary")
                    clear_kb_btn = gr.Button("Clear Knowledge Base", variant="secondary")
                
                upload_status = gr.Textbox(
                    label="Upload/Reload Status",
                    value="Ready to load abstracts...",
                    interactive=False,
                    lines=3
                )
                
                knowledge_status = gr.Textbox(
                    label="Knowledge Base Status",
                    value="📚 Loading abstracts on startup...",
                    interactive=False,
                    lines=1
                )
            
        with gr.Column(scale=1):
            # Settings panel
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")
                
                enable_rag = gr.Checkbox(
                    label="Enable RAG",
                    value=True,
                    info="Use uploaded documents for context"
                )
                
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="You are an emergency medicine expert...",
                    lines=3,
                    value="You are an expert emergency medicine physician with extensive knowledge of clinical protocols, evidence-based practices, and emergency care guidelines. Provide accurate, practical, and safety-focused medical information."
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
                
                if USE_OPENAI and MODEL_NAME in ["o4-mini-2025-04-16", "o3-2025-04-16"]:
                    gr.Markdown("**Temperature:** ⚠️ *This OpenAI model uses default temperature (1.0) only*")
                    temp_info = "Note: This OpenAI model ignores temperature setting"
                elif USE_OPENAI:
                    gr.Markdown("**Temperature:** Factual (medical) ← → Creative")
                    temp_info = "Controls randomness in responses"
                else:
                    gr.Markdown("**Temperature:** Factual (medical) ← → Creative")
                    temp_info = "Controls randomness in responses"
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.1,
                    step=0.1,
                    label="Temperature",
                    info=temp_info
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
                if USE_OPENAI:
                    models_btn = gr.Button("Show Available Models", variant="secondary")
            
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
        fn=reload_abstracts,
        inputs=[],
        outputs=[upload_status, knowledge_status]
    )
    
    def save_temp_pdfs():
        """Save temporary PDFs and return status"""
        saved_files = save_temp_pdfs_to_abstracts()
        if saved_files:
            status = f"📁 **Saved {len(saved_files)} PDFs to abstracts/:**\n" + "\n".join([f"✅ {f}" for f in saved_files])
            status += f"\n\n💡 These PDFs will now auto-load on next startup."
        else:
            status = "ℹ️  No temporary PDFs found to save."
        
        knowledge_status = f"📚 Knowledge Base: {len(document_store)} chunks from {len(set(chunk['source'] for chunk in document_store))} documents"
        return status, knowledge_status
    
    save_temp_btn.click(
        fn=save_temp_pdfs,
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
    
    # Add models button click handler only if using OpenAI
    if USE_OPENAI:
        models_btn.click(
            fn=display_available_models,
            outputs=stats_display
        )

if __name__ == "__main__":
    print("🚀 Starting Emergency Medicine RAG Chat Interface...")
    print(f"🤖 AI Provider: {'☁️  OpenAI' if USE_OPENAI else '🏠 Local AI (LM Studio)'}")
    print(f"📡 API Endpoint: {'OpenAI API' if USE_OPENAI else LM_STUDIO_BASE_URL}")
    print(f"🤖 LLM Model: {MODEL_NAME}")
    print(f"🔍 Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"📄 Chunk Size: {CHUNK_SIZE}")
    print(f"🎯 Max Relevant Chunks: {MAX_RELEVANT_CHUNKS}")
    print(f"📊 Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"📁 Abstracts Directory: ./abstracts")
    
    # Show current knowledge base status
    if document_store:
        sources = list(set(chunk['source'] for chunk in document_store))
        print(f"📚 Knowledge Base: {len(document_store)} chunks from {len(sources)} documents")
    else:
        print("📚 Knowledge Base: Empty (no abstracts loaded)")
    
    # Check API health at startup
    print("🔍 Checking API connections...")
    api_name = "OpenAI" if USE_OPENAI else "LM Studio LLM"
    if check_api_health():
        print(f"✅ {api_name} API is accessible")
    else:
        print(f"❌ Warning: Cannot connect to {api_name} API")
    
    embedding_api_name = "OpenAI" if USE_OPENAI else "LM Studio Embedding"
    if check_embedding_api_health():
        print(f"✅ {embedding_api_name} API is accessible")
    else:
        print(f"❌ Warning: Cannot connect to {embedding_api_name} API")
    
    print("\n" + "="*50)
    print("🌐 Starting web interface...")
    print("📱 Access at: http://localhost:7866")
    print("🔄 To reload abstracts, use the 'Reload from abstracts directory' button")
    print("=" * 50)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7866,  # Different port
        share=False,
        show_error=True
    )

def save_temp_pdfs_to_abstracts():
    """Save any remaining temporary PDFs to abstracts directory"""
    import glob
    import shutil
    
    # Ensure abstracts directory exists
    abstracts_dir = "abstracts"
    os.makedirs(abstracts_dir, exist_ok=True)
    
    # Look for temporary Gradio PDFs
    temp_patterns = [
        "/tmp/gradio/*/*.pdf",
        "/tmp/*gradio*/*.pdf",
        "/tmp/*/gradio*/*.pdf"
    ]
    
    saved_files = []
    
    for pattern in temp_patterns:
        temp_files = glob.glob(pattern)
        for temp_file in temp_files:
            try:
                file_name = os.path.basename(temp_file)
                abstracts_path = os.path.join(abstracts_dir, file_name)
                
                # Avoid overwriting existing files
                counter = 1
                original_name = file_name
                while os.path.exists(abstracts_path):
                    name_parts = original_name.rsplit('.', 1)
                    if len(name_parts) == 2:
                        file_name = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    else:
                        file_name = f"{original_name}_{counter}"
                    abstracts_path = os.path.join(abstracts_dir, file_name)
                    counter += 1
                
                # Copy file to permanent location
                shutil.copy2(temp_file, abstracts_path)
                saved_files.append(file_name)
                print(f"📁 Saved {file_name} to abstracts directory")
                
            except Exception as e:
                print(f"❌ Failed to save {temp_file}: {e}")
    
    return saved_files
