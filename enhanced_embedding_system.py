"""
Enhanced embedding system for emergency_rag_chatbot.py
This allows switching between different embedding models while keeping
the existing LM Studio integration.
"""

import requests
import numpy as np
from typing import List, Optional, Dict
import os
import time

class MedicalEmbeddingManager:
    """
    Drop-in replacement for the existing embedding system
    that supports both LM Studio and local medical models.
    """
    
    def __init__(self, lm_studio_url: str = "http://10.5.0.2:1234"):
        self.lm_studio_url = lm_studio_url
        self.current_model_type = "lm_studio"  # Default to current setup
        self.current_model_name = "text-embedding-all-minilm-l6-v2-embedding"
        self.local_model = None
        
        # Available medical models
        self.available_models = {
            "lm_studio_minilm": {
                "type": "lm_studio",
                "name": "text-embedding-all-minilm-l6-v2-embedding",
                "description": "Current MiniLM (baseline)"
            },
            "clinical_bert": {
                "type": "local",
                "name": "emilyalsentzer/Bio_ClinicalBERT",
                "description": "Clinical BERT - Best for clinical notes and emergency medicine"
            },
            "pubmed_bert": {
                "type": "local", 
                "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                "description": "PubMed BERT - Best for medical research papers"
            },
            "sapbert": {
                "type": "local",
                "name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", 
                "description": "SapBERT - Best for medical entity understanding"
            },
            "bge_large": {
                "type": "local",
                "name": "BAAI/bge-large-en-v1.5",
                "description": "BGE Large - High performance general model"
            }
        }
    
    def switch_model(self, model_key: str) -> bool:
        """Switch to a different embedding model"""
        
        if model_key not in self.available_models:
            print(f"❌ Unknown model: {model_key}")
            print(f"Available models: {list(self.available_models.keys())}")
            return False
        
        model_config = self.available_models[model_key]
        
        if model_config["type"] == "lm_studio":
            self.current_model_type = "lm_studio"
            self.current_model_name = model_config["name"]
            print(f"✅ Switched to LM Studio model: {model_config['description']}")
            return True
            
        elif model_config["type"] == "local":
            try:
                from sentence_transformers import SentenceTransformer
                print(f"🔄 Loading {model_config['description']}...")
                self.local_model = SentenceTransformer(model_config["name"])
                self.current_model_type = "local"
                self.current_model_name = model_config["name"]
                print(f"✅ Loaded: {model_config['description']}")
                return True
            except ImportError:
                print("❌ sentence-transformers not installed")
                print("Install with: pip install sentence-transformers")
                return False
            except Exception as e:
                print(f"❌ Failed to load {model_config['name']}: {e}")
                return False
        
        return False
    
    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode texts using current model (drop-in replacement for call_embedding_api)"""
        
        if self.current_model_type == "lm_studio":
            return self._encode_lm_studio(texts)
        elif self.current_model_type == "local":
            return self._encode_local(texts)
        else:
            print("❌ No model configured")
            return None
    
    def _encode_lm_studio(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode using LM Studio (existing functionality)"""
        url = f"{self.lm_studio_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.current_model_name,
            "input": texts
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            embeddings = []
            
            for item in data.get("data", []):
                embeddings.append(item["embedding"])
            
            return np.array(embeddings)
            
        except Exception as e:
            print(f"❌ LM Studio Embedding Error: {e}")
            return None
    
    def _encode_local(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode using local model"""
        try:
            if self.local_model is None:
                print("❌ No local model loaded")
                return None
                
            embeddings = self.local_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"❌ Local Embedding Error: {e}")
            return None
    
    def get_current_model_info(self) -> Dict:
        """Get information about current model"""
        for key, config in self.available_models.items():
            if config["name"] == self.current_model_name:
                return {
                    "key": key,
                    "type": self.current_model_type,
                    "name": self.current_model_name,
                    "description": config["description"]
                }
        
        return {
            "key": "unknown",
            "type": self.current_model_type, 
            "name": self.current_model_name,
            "description": "Unknown model"
        }
    
    def benchmark_current_model(self) -> Dict:
        """Benchmark current model with medical queries"""
        test_queries = [
            "anaphylaxis treatment emergency",
            "cardiac arrest ACLS protocol", 
            "sepsis diagnosis criteria",
            "stroke tPA administration",
            "chest pain differential diagnosis"
        ]
        
        print(f"🔍 Benchmarking current model...")
        start_time = time.time()
        
        embeddings = self.encode(test_queries)
        
        end_time = time.time()
        
        if embeddings is None:
            return {"success": False, "error": "Failed to generate embeddings"}
        
        total_time = end_time - start_time
        
        return {
            "success": True,
            "total_time": total_time,
            "avg_time_per_query": total_time / len(test_queries),
            "queries_per_second": len(test_queries) / total_time,
            "embedding_dimension": embeddings.shape[1],
            "model_info": self.get_current_model_info()
        }

# Integration example for existing emergency_rag_chatbot.py
def create_enhanced_embedding_system():
    """
    Create enhanced embedding system that can be integrated 
    into existing emergency_rag_chatbot.py
    """
    
    embedding_manager = MedicalEmbeddingManager()
    
    def enhanced_call_embedding_api(texts: List[str]) -> Optional[np.ndarray]:
        """Enhanced version of call_embedding_api function"""
        return embedding_manager.encode(texts)
    
    def switch_embedding_model(model_key: str) -> bool:
        """Switch embedding model"""
        return embedding_manager.switch_model(model_key)
    
    def get_embedding_model_info() -> Dict:
        """Get current embedding model info"""
        return embedding_manager.get_current_model_info()
    
    def benchmark_embedding_model() -> Dict:
        """Benchmark current embedding model"""
        return embedding_manager.benchmark_current_model()
    
    return {
        "call_embedding_api": enhanced_call_embedding_api,
        "switch_model": switch_embedding_model,
        "get_model_info": get_embedding_model_info,
        "benchmark_model": benchmark_embedding_model,
        "available_models": embedding_manager.available_models
    }

# Usage example
if __name__ == "__main__":
    print("🏥 Medical Embedding System")
    print("=" * 40)
    
    # Create enhanced system
    embedding_system = create_enhanced_embedding_system()
    
    print("\n📋 Available Models:")
    for key, model in embedding_system["available_models"].items():
        print(f"  {key}: {model['description']}")
    
    print(f"\n🔄 Current model: {embedding_system['get_model_info']()['description']}")
    
    # Benchmark current model
    benchmark = embedding_system["benchmark_model"]()
    if benchmark["success"]:
        print(f"\n📊 Performance:")
        print(f"  ⏱️  Avg time per query: {benchmark['avg_time_per_query']:.3f}s")
        print(f"  🚀 Queries per second: {benchmark['queries_per_second']:.1f}")
        print(f"  📐 Embedding dimension: {benchmark['embedding_dimension']}")
    
    print(f"\n💡 To switch models:")
    print(f"   embedding_system['switch_model']('clinical_bert')")
    print(f"   embedding_system['switch_model']('pubmed_bert')")
    print(f"   embedding_system['switch_model']('sapbert')")
