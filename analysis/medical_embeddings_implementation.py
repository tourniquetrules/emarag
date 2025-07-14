"""
Flexible embedding implementation supporting multiple medical embedding models.
This allows easy switching between different embedding models for comparison.
"""

import os
import requests
import numpy as np
from typing import List, Optional
import time

class EmbeddingModelManager:
    """Manages different embedding models for medical RAG system"""
    
    def __init__(self):
        self.current_model = None
        self.model_type = None
        self.api_base_url = "http://10.5.0.2:1234"
        
    def set_lm_studio_model(self, model_name: str):
        """Set LM Studio embedding model (current approach)"""
        self.current_model = model_name
        self.model_type = "lm_studio"
        print(f"🔄 Set LM Studio embedding model: {model_name}")
        
    def set_huggingface_model(self, model_name: str):
        """Set HuggingFace embedding model for local inference"""
        try:
            from sentence_transformers import SentenceTransformer
            self.current_model = SentenceTransformer(model_name)
            self.model_type = "huggingface"
            print(f"✅ Loaded HuggingFace model: {model_name}")
        except ImportError:
            print("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
            return False
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return False
        return True
            
    def set_openai_model(self, model_name: str, api_key: str):
        """Set OpenAI embedding model"""
        try:
            import openai
            openai.api_key = api_key
            self.current_model = model_name
            self.model_type = "openai"
            print(f"✅ Set OpenAI embedding model: {model_name}")
        except ImportError:
            print("❌ openai not installed. Run: pip install openai")
            return False
        return True
    
    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode texts using current embedding model"""
        
        if self.model_type == "lm_studio":
            return self._encode_lm_studio(texts)
        elif self.model_type == "huggingface":
            return self._encode_huggingface(texts)
        elif self.model_type == "openai":
            return self._encode_openai(texts)
        else:
            print("❌ No embedding model set")
            return None
    
    def _encode_lm_studio(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode using LM Studio API"""
        url = f"{self.api_base_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.current_model,
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
    
    def _encode_huggingface(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode using HuggingFace model"""
        try:
            embeddings = self.current_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"❌ HuggingFace Embedding Error: {e}")
            return None
    
    def _encode_openai(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode using OpenAI API"""
        try:
            import openai
            
            # Handle batch size limits
            batch_size = 100  # OpenAI limit
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = openai.Embedding.create(
                    model=self.current_model,
                    input=batch
                )
                
                batch_embeddings = [item["embedding"] for item in response["data"]]
                all_embeddings.extend(batch_embeddings)
                
                time.sleep(0.1)  # Rate limiting
            
            return np.array(all_embeddings)
            
        except Exception as e:
            print(f"❌ OpenAI Embedding Error: {e}")
            return None
    
    def benchmark_model(self, test_queries: List[str], model_name: str = None) -> dict:
        """Benchmark current embedding model"""
        if model_name:
            print(f"🔍 Benchmarking: {model_name}")
        
        start_time = time.time()
        embeddings = self.encode(test_queries)
        end_time = time.time()
        
        if embeddings is None:
            return {"success": False, "error": "Failed to generate embeddings"}
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_query = total_time / len(test_queries)
        embedding_dimension = embeddings.shape[1]
        
        return {
            "success": True,
            "total_time": total_time,
            "avg_time_per_query": avg_time_per_query,
            "queries_per_second": len(test_queries) / total_time,
            "embedding_dimension": embedding_dimension,
            "model_type": self.model_type,
            "model_name": str(self.current_model)[:50] + "..." if len(str(self.current_model)) > 50 else str(self.current_model)
        }

# Medical embedding model configurations
MEDICAL_EMBEDDING_CONFIGS = {
    "current_minilm": {
        "type": "lm_studio",
        "model": "text-embedding-all-minilm-l6-v2-embedding",
        "description": "Current MiniLM model (baseline)"
    },
    
    "clinical_bert": {
        "type": "huggingface", 
        "model": "emilyalsentzer/Bio_ClinicalBERT",
        "description": "Clinical BERT - trained on clinical notes"
    },
    
    "pubmed_bert": {
        "type": "huggingface",
        "model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
        "description": "PubMed BERT - trained on medical research"
    },
    
    "sapbert": {
        "type": "huggingface",
        "model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "description": "SapBERT - medical entity understanding"
    },
    
    "medical_sentence_bert": {
        "type": "huggingface",
        "model": "pritamdeka/S-PubMedBert-MS-MARCO",
        "description": "Medical Sentence-BERT"
    },
    
    "bge_large": {
        "type": "huggingface", 
        "model": "BAAI/bge-large-en-v1.5",
        "description": "BGE Large - high performance general model"
    },
    
    "e5_large": {
        "type": "huggingface",
        "model": "intfloat/e5-large-v2", 
        "description": "E5 Large - state-of-the-art general model"
    }
}

def compare_embedding_models():
    """Compare different embedding models on medical queries"""
    
    # Test queries representative of emergency medicine
    test_queries = [
        "anaphylaxis treatment protocol emergency department",
        "cardiac arrest ACLS guidelines resuscitation",
        "sepsis diagnosis criteria emergency medicine",
        "stroke tPA administration time window",
        "chest pain differential diagnosis emergency",
        "pediatric fever management emergency department",
        "trauma assessment primary survey ATLS",
        "respiratory distress ventilation emergency"
    ]
    
    print("🏥 Medical Embedding Model Comparison")
    print("=" * 60)
    
    manager = EmbeddingModelManager()
    results = []
    
    for config_name, config in MEDICAL_EMBEDDING_CONFIGS.items():
        print(f"\n🔍 Testing: {config['description']}")
        
        # Set up the model
        success = False
        if config["type"] == "lm_studio":
            manager.set_lm_studio_model(config["model"])
            success = True
        elif config["type"] == "huggingface":
            success = manager.set_huggingface_model(config["model"])
        
        if not success:
            print(f"❌ Skipping {config_name} - setup failed")
            continue
        
        # Benchmark the model
        benchmark = manager.benchmark_model(test_queries, config_name)
        
        if benchmark["success"]:
            results.append({
                "name": config_name,
                "description": config["description"],
                **benchmark
            })
            
            print(f"   ✅ Success!")
            print(f"   ⏱️  Average time per query: {benchmark['avg_time_per_query']:.3f}s")
            print(f"   🚀 Queries per second: {benchmark['queries_per_second']:.1f}")
            print(f"   📐 Embedding dimension: {benchmark['embedding_dimension']}")
        else:
            print(f"   ❌ Failed: {benchmark.get('error', 'Unknown error')}")
    
    # Summary table
    if results:
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Model':<25} {'Speed (q/s)':<12} {'Avg Time':<12} {'Dimension':<12}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['name']:<25} {result['queries_per_second']:<12.1f} "
                  f"{result['avg_time_per_query']:<12.3f} {result['embedding_dimension']:<12}")
    
    return results

if __name__ == "__main__":
    # Install required packages
    print("📦 To use medical embeddings, install required packages:")
    print("pip install sentence-transformers transformers torch")
    print("\n🔄 Run comparison:")
    print("python medical_embeddings_implementation.py")
    
    # Uncomment to run comparison
    # compare_embedding_models()
