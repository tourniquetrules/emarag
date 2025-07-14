#!/usr/bin/env python3
"""
Script to rebuild the emergency medicine RAG knowledge base cleanly
"""

import os
import sys
import requests

def check_lm_studio_health():
    """Check if LM Studio is running"""
    try:
        response = requests.get("http://10.5.0.2:1234/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🔧 Emergency Medicine RAG Knowledge Base Rebuild Tool")
    print("=" * 60)
    
    # Check LM Studio
    if not check_lm_studio_health():
        print("❌ LM Studio API not accessible at http://10.5.0.2:1234")
        print("   Please start LM Studio and ensure the API is running")
        sys.exit(1)
    
    print("✅ LM Studio API is accessible")
    
    # Check abstracts directory
    abstracts_dir = "abstracts"
    if not os.path.exists(abstracts_dir):
        print(f"❌ Abstracts directory '{abstracts_dir}' not found")
        sys.exit(1)
    
    pdf_files = [f for f in os.listdir(abstracts_dir) if f.endswith('.pdf')]
    print(f"📁 Found {len(pdf_files)} PDF files in abstracts directory")
    
    print("\n🚀 Starting emergency medicine RAG chatbot with clean rebuild...")
    print("   This will:")
    print("   1. Clear any existing knowledge base")
    print("   2. Rebuild embeddings from scratch")
    print("   3. Start the web interface")
    print("\n   Please wait for the rebuild to complete...")
    
    # Start the chatbot
    os.system("python3 emergency_rag_chatbot.py")

if __name__ == "__main__":
    main()
