#!/bin/bash

# Medical Embeddings Setup Script
echo "🏥 Setting up Medical Embeddings for Emergency Medicine RAG"
echo "============================================================"

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    echo "✅ Using mamba for faster installation"
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    echo "✅ Using conda"
    CONDA_CMD="conda"
else
    echo "⚠️  No conda found, using pip"
    CONDA_CMD="pip"
fi

# Install required packages
echo "📦 Installing required packages..."

if [ "$CONDA_CMD" != "pip" ]; then
    # Using conda/mamba
    $CONDA_CMD install -c conda-forge transformers sentence-transformers torch -y
    pip install faiss-cpu  # faiss not always available in conda
else
    # Using pip only
    pip install transformers sentence-transformers torch faiss-cpu
fi

echo "✅ Base packages installed"

# Download and cache medical models
echo "🔄 Pre-downloading medical embedding models..."
echo "This may take a few minutes for the first time..."

python3 << 'EOF'
import os
from sentence_transformers import SentenceTransformer

# Models to pre-download
models = [
    "emilyalsentzer/Bio_ClinicalBERT",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "BAAI/bge-large-en-v1.5"
]

for model_name in models:
    try:
        print(f"📥 Downloading {model_name}...")
        model = SentenceTransformer(model_name)
        print(f"✅ Cached {model_name}")
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        
print("\n🎉 Medical embedding models are ready!")
EOF

echo ""
echo "🚀 Setup complete! You can now:"
echo "1. Test models: python3 test_medical_embeddings.py"
echo "2. Compare performance: python3 medical_embeddings_implementation.py"
echo "3. Integrate with RAG: See enhanced_embedding_system.py"
