#!/bin/bash

# Setup script for Emergency Medicine RAG Chatbot
# This script fixes PATH warnings and sets up the environment

echo "🚀 Setting up Emergency Medicine RAG Chatbot..."

# Add ~/.local/bin to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "📝 Adding ~/.local/bin to PATH..."
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ PATH updated"
else
    echo "✅ ~/.local/bin already in PATH"
fi

# Install requirements
echo "📦 Installing Python dependencies..."
pip install --user -r requirements.txt

# Create abstracts directory if it doesn't exist
if [ ! -d "abstracts" ]; then
    echo "📁 Creating abstracts directory..."
    mkdir abstracts
    echo "ℹ️  Place your emergency medicine PDF abstracts in the 'abstracts' directory"
else
    echo "✅ Abstracts directory exists"
    pdf_count=$(find abstracts -name "*.pdf" | wc -l)
    echo "📄 Found $pdf_count PDF files in abstracts directory"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the Emergency Medicine RAG Chatbot:"
echo "  python emergency_rag_chatbot.py"
echo ""
echo "The system will automatically load PDFs from the 'abstracts' directory on startup."
echo "You can also upload additional PDFs through the web interface."
echo ""
