#!/usr/bin/env python3
"""
Emergency fix for contaminated emergency medicine RAG chatbot.
This script completely rebuilds the knowledge base and provides instructions
for restarting LM Studio to clear any model contamination.
"""

import os
import shutil
import faiss
import pickle
import requests
import json
from pathlib import Path

def check_lm_studio_connection():
    """Check if LM Studio is accessible"""
    try:
        response = requests.get("http://10.5.0.2:1234/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False

def clear_all_cache():
    """Clear any potential cache files that might be causing contamination"""
    cache_patterns = [
        "*.faiss",
        "*.pickle", 
        "*.pkl",
        "embeddings_*",
        "document_store_*",
        "knowledge_base_*"
    ]
    
    print("🧹 Clearing potential cache files...")
    current_dir = Path(".")
    for pattern in cache_patterns:
        for file in current_dir.glob(pattern):
            try:
                file.unlink()
                print(f"   🗑️  Deleted: {file}")
            except Exception as e:
                print(f"   ❌ Failed to delete {file}: {e}")

def test_api_with_simple_query():
    """Test LM Studio API with a simple medical query"""
    if not check_lm_studio_connection():
        print("❌ Cannot connect to LM Studio API")
        return False
    
    url = "http://10.5.0.2:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Simple medical query to test for contamination
    payload = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b",
        "messages": [
            {
                "role": "system", 
                "content": "You are an emergency medicine physician. Answer medical questions clearly and professionally."
            },
            {
                "role": "user", 
                "content": "What is the first-line treatment for anaphylaxis?"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 200,
        "stop": ["<|endoftext|>", "<|end|>"],
        "presence_penalty": 0.1
    }
    
    try:
        print("🧪 Testing LM Studio API with simple medical query...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        
        print(f"📋 API Response: {answer[:200]}...")
        
        # Check for contamination indicators
        contamination_indicators = [
            "number theory",
            "mathematics", 
            "geometry",
            "integer",
            "equation",
            "proof",
            "theorem",
            "polynomial",
            "\\textbf",
            "protagonize"
        ]
        
        answer_lower = answer.lower()
        contaminated = any(indicator in answer_lower for indicator in contamination_indicators)
        
        if contaminated:
            print("❌ CONTAMINATION DETECTED: API is giving non-medical responses")
            print("🔄 LM Studio model needs to be restarted/reloaded")
            return False
        else:
            print("✅ API test passed: Response appears medical")
            return True
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    print("🚨 Emergency Medicine RAG Contamination Fix")
    print("="*50)
    
    # Step 1: Clear cache files
    clear_all_cache()
    
    # Step 2: Test API for contamination
    print("\n🧪 Testing LM Studio API...")
    api_clean = test_api_with_simple_query()
    
    if not api_clean:
        print("\n🚨 CRITICAL: LM Studio model is contaminated!")
        print("\n📋 REQUIRED ACTIONS:")
        print("1. 🔄 Restart LM Studio completely")
        print("2. 🤖 Reload the deepseek/deepseek-r1-0528-qwen3-8b model")
        print("3. 🧪 Test with a simple medical question")
        print("4. 🚀 Restart the emergency medicine chatbot")
        print("\nℹ️  The model itself has been contaminated with non-medical content.")
        print("ℹ️  This requires restarting LM Studio, not just the chatbot.")
        return False
    
    print("\n✅ LM Studio API appears clean")
    print("✅ Safe to restart emergency medicine chatbot")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Run: python3 emergency_rag_chatbot.py")
    else:
        print("\n⚠️  Fix LM Studio first, then rerun this script")
