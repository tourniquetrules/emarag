#!/bin/bash
# Auto-start script for Emergency Medicine RAG Chatbot

cd /home/vboxuser/emarag
source venv312/bin/activate

# Send "1" to select Local AI automatically
echo "1" | python emergency_rag_chatbot.py
