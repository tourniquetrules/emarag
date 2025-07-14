"""
Medical Embedding Models Comparison and Implementation Guide
==========================================================

This file contains information about medical-specific embedding models
that are better suited for emergency medicine documentation.
"""

# Medical-Specific Embedding Models (Recommended)
MEDICAL_EMBEDDING_OPTIONS = {
    "clinical_bert": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "description": "BERT trained on clinical notes and biomedical literature",
        "advantages": ["Clinical terminology", "Medical abbreviations", "Disease entities"],
        "size": "110M parameters",
        "implementation": "huggingface_transformers"
    },
    
    "pubmed_bert": {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "description": "BERT pre-trained on PubMed abstracts and full-text articles",
        "advantages": ["Medical research terminology", "Scientific language", "Drug names"],
        "size": "110M parameters", 
        "implementation": "huggingface_transformers"
    },
    
    "biogpt_embeddings": {
        "model_name": "microsoft/biogpt",
        "description": "Generative model trained on biomedical literature with embedding capability",
        "advantages": ["Latest medical knowledge", "Research paper understanding", "Medical reasoning"],
        "size": "1.5B parameters",
        "implementation": "huggingface_transformers"
    },
    
    "sapbert": {
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "description": "Medical entity linking and synonym recognition",
        "advantages": ["Medical entity understanding", "Synonym detection", "UMLS concepts"],
        "size": "110M parameters",
        "implementation": "huggingface_transformers"
    },
    
    "medical_sentence_transformers": {
        "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
        "description": "Sentence-BERT fine-tuned on medical literature",
        "advantages": ["Sentence-level understanding", "Medical document similarity", "Fast inference"],
        "size": "110M parameters",
        "implementation": "sentence_transformers"
    }
}

# High-Performance General Models (Also good for medical)
GENERAL_EMBEDDING_OPTIONS = {
    "e5_large": {
        "model_name": "intfloat/e5-large-v2",
        "description": "State-of-the-art general embedding model",
        "advantages": ["Excellent general performance", "Good medical performance", "Fast"],
        "size": "335M parameters",
        "implementation": "huggingface_transformers"
    },
    
    "bge_large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "description": "Beijing Academy AI general embedding model",
        "advantages": ["Top performance", "Good domain transfer", "Efficient"],
        "size": "335M parameters",
        "implementation": "huggingface_transformers"
    },
    
    "instructor_xl": {
        "model_name": "hkunlp/instructor-xl",
        "description": "Instruction-tuned embedding model",
        "advantages": ["Task-specific instructions", "Flexible", "High quality"],
        "size": "1.3B parameters",
        "implementation": "instructor_embeddings"
    }
}

# Performance Comparison (approximate)
EMBEDDING_PERFORMANCE = {
    "text-embedding-all-minilm-l6-v2": {
        "medical_score": 6.5,  # Current model
        "speed": "very_fast",
        "size": "small"
    },
    "Bio_ClinicalBERT": {
        "medical_score": 8.5,
        "speed": "fast", 
        "size": "medium"
    },
    "PubMedBERT": {
        "medical_score": 8.8,
        "speed": "fast",
        "size": "medium"
    },
    "SapBERT": {
        "medical_score": 9.0,
        "speed": "fast",
        "size": "medium"
    },
    "e5-large-v2": {
        "medical_score": 8.0,
        "speed": "medium",
        "size": "large"
    },
    "bge-large-en-v1.5": {
        "medical_score": 8.2,
        "speed": "medium", 
        "size": "large"
    }
}

print("Medical Embedding Models Analysis")
print("=" * 50)
print("\n🏥 MEDICAL-SPECIFIC MODELS:")
for name, info in MEDICAL_EMBEDDING_OPTIONS.items():
    print(f"\n📋 {name.upper()}:")
    print(f"   Model: {info['model_name']}")
    print(f"   Description: {info['description']}")
    print(f"   Advantages: {', '.join(info['advantages'])}")
    print(f"   Size: {info['size']}")

print("\n\n🔬 GENERAL HIGH-PERFORMANCE MODELS:")
for name, info in GENERAL_EMBEDDING_OPTIONS.items():
    print(f"\n📊 {name.upper()}:")
    print(f"   Model: {info['model_name']}")
    print(f"   Description: {info['description']}")
    print(f"   Advantages: {', '.join(info['advantages'])}")
    print(f"   Size: {info['size']}")

print("\n\n📈 RECOMMENDED FOR EMERGENCY MEDICINE:")
print("1. 🥇 SapBERT (Best medical understanding)")
print("2. 🥈 PubMedBERT (Strong research paper comprehension)")  
print("3. 🥉 Bio_ClinicalBERT (Good clinical terminology)")
print("4. 🏆 BGE-Large (Best general performance with good medical transfer)")
