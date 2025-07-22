#!/bin/bash
# Auto-start script for Emergency Medicine RAG Chatbot

cd /home/tourniquetrules/emarag
source venv312/bin/activate

echo "Emergency Medicine RAG Chatbot Startup"
echo "======================================"
echo "1. Local AI (LM Studio)"
echo "2. OpenAI"
echo ""
read -p "Select AI provider (1 or 2): " choice

echo "Starting chatbot in background..."

# Send the choice to the python script and run in background with nohup
echo "$choice" | nohup python emergency_rag_chatbot.py > chatbot.log 2>&1 &

echo "Chatbot started in background!"
echo "- Log file: chatbot.log"
echo "- Process ID: $!"
echo "- Access at: http://localhost:7866"
echo "- Public URL: https://emachatbot.haydd.com"
echo ""
echo "To stop the chatbot, run: kill $!"
