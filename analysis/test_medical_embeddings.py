#!/usr/bin/env python3
"""
Test script for medical embedding models.
Tests different models on emergency medicine queries and compares performance.
"""

import time
import numpy as np
from enhanced_embedding_system import MedicalEmbeddingManager

def test_medical_embeddings():
    """Test different medical embedding models"""
    
    # Emergency medicine test queries
    test_queries = [
        "anaphylaxis treatment epinephrine emergency department",
        "cardiac arrest ACLS guidelines chest compressions",
        "sepsis diagnosis lactate blood pressure", 
        "stroke tPA thrombolysis time window",
        "chest pain myocardial infarction ECG",
        "pediatric fever management antibiotics",
        "trauma assessment ATLS primary survey",
        "respiratory distress intubation ventilation",
        "diabetic ketoacidosis insulin protocol",
        "seizure status epilepticus benzodiazepines"
    ]
    
    print("🏥 Medical Embedding Model Testing")
    print("=" * 50)
    print(f"📝 Testing with {len(test_queries)} emergency medicine queries")
    
    manager = MedicalEmbeddingManager()
    results = []
    
    # Test each available model
    for model_key, model_config in manager.available_models.items():
        print(f"\n🔍 Testing: {model_config['description']}")
        print(f"   Model: {model_config['name']}")
        
        # Switch to model
        if not manager.switch_model(model_key):
            print(f"   ❌ Skipping - failed to load")
            continue
        
        # Benchmark
        start_time = time.time()
        embeddings = manager.encode(test_queries)
        end_time = time.time()
        
        if embeddings is None:
            print(f"   ❌ Failed to generate embeddings")
            continue
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / len(test_queries)
        qps = len(test_queries) / total_time
        dimension = embeddings.shape[1]
        
        result = {
            "model_key": model_key,
            "description": model_config['description'],
            "type": model_config['type'],
            "total_time": total_time,
            "avg_time": avg_time,
            "qps": qps,
            "dimension": dimension,
            "embeddings": embeddings
        }
        results.append(result)
        
        print(f"   ✅ Success!")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   📊 Avg per query: {avg_time:.3f}s")
        print(f"   🚀 Queries/sec: {qps:.1f}")
        print(f"   📐 Dimension: {dimension}")
    
    return results, test_queries

def analyze_embeddings(results, test_queries):
    """Analyze embedding quality through similarity comparisons"""
    
    if len(results) < 2:
        print("⚠️  Need at least 2 models to compare")
        return
    
    print(f"\n🔬 Embedding Quality Analysis")
    print("=" * 50)
    
    # Test semantic similarity for medical concepts
    medical_pairs = [
        ("anaphylaxis treatment epinephrine", "severe allergic reaction emergency"),
        ("cardiac arrest ACLS", "heart stopped resuscitation"),
        ("sepsis diagnosis criteria", "systemic infection response"),
        ("stroke tPA thrombolysis", "brain attack clot buster"),
        ("chest pain myocardial infarction", "heart attack symptoms")
    ]
    
    print("📊 Semantic Similarity Scores (medical concept pairs):")
    print("Higher scores = better medical understanding")
    print()
    
    for result in results:
        model_name = result['model_key']
        embeddings = result['embeddings']
        
        similarities = []
        
        for query1, query2 in medical_pairs:
            # Find embeddings for the pair
            try:
                # Simple approach: encode the pair directly
                manager = MedicalEmbeddingManager()
                manager.switch_model(model_name)
                
                pair_embeddings = manager.encode([query1, query2])
                if pair_embeddings is not None:
                    # Calculate cosine similarity
                    emb1 = pair_embeddings[0]
                    emb2 = pair_embeddings[1]
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append(similarity)
                    
            except Exception as e:
                print(f"   ❌ Error calculating similarity: {e}")
                similarities.append(0.0)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        print(f"🏥 {model_name:<20} | Avg similarity: {avg_similarity:.3f}")
        for i, (q1, q2) in enumerate(medical_pairs):
            if i < len(similarities):
                print(f"   {q1[:25]:<25} ↔ {q2[:25]:<25} : {similarities[i]:.3f}")

def print_summary(results):
    """Print performance summary"""
    
    if not results:
        print("❌ No results to summarize")
        return
    
    print(f"\n📈 Performance Summary")
    print("=" * 70)
    print(f"{'Model':<25} {'Type':<10} {'Speed (q/s)':<12} {'Avg Time':<12} {'Dimension':<12}")
    print("-" * 70)
    
    # Sort by speed
    sorted_results = sorted(results, key=lambda x: x['qps'], reverse=True)
    
    for result in sorted_results:
        print(f"{result['model_key']:<25} {result['type']:<10} {result['qps']:<12.1f} "
              f"{result['avg_time']:<12.3f} {result['dimension']:<12}")
    
    # Recommendations
    print(f"\n🏆 Recommendations for Emergency Medicine:")
    
    fastest = max(results, key=lambda x: x['qps'])
    print(f"🚀 Fastest: {fastest['model_key']} ({fastest['qps']:.1f} q/s)")
    
    medical_models = [r for r in results if 'bert' in r['model_key'].lower() or 'clinical' in r['model_key'].lower()]
    if medical_models:
        best_medical = max(medical_models, key=lambda x: x['qps'])
        print(f"🏥 Best Medical: {best_medical['model_key']} ({best_medical['qps']:.1f} q/s)")
    
    largest_dim = max(results, key=lambda x: x['dimension'])
    print(f"📐 Highest Dimension: {largest_dim['model_key']} ({largest_dim['dimension']} dims)")

def main():
    """Main testing function"""
    
    print("🚀 Starting medical embedding tests...")
    print("This will test available models and compare their performance.\n")
    
    try:
        # Run tests
        results, test_queries = test_medical_embeddings()
        
        if not results:
            print("❌ No models could be tested")
            print("💡 Try installing models first: bash setup_medical_embeddings.sh")
            return
        
        # Analyze results
        analyze_embeddings(results, test_queries)
        print_summary(results)
        
        # Final recommendations
        print(f"\n💡 Next Steps:")
        print("1. 🏥 For best medical understanding: Use clinical_bert or pubmed_bert")
        print("2. 🚀 For best speed: Use current LM Studio model")  
        print("3. ⚖️  For balanced performance: Use bge_large")
        print("4. 🔬 To integrate: See enhanced_embedding_system.py")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
