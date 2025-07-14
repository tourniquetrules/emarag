"""
Enhanced conversation management for multiple RAG systems.
Each system maintains separate conversation state and context.
"""

class RAGSessionManager:
    def __init__(self, session_name, model_config):
        self.session_name = session_name
        self.model_config = model_config
        self.conversation_id = f"{session_name}_{int(time.time())}"
        self.conversation_history = []
        self.context_limit = 50  # Maximum message pairs to keep
        
    def create_isolated_payload(self, messages, user_message):
        """Create API payload with session isolation"""
        
        # Clear context boundary marker
        boundary_message = {
            "role": "system", 
            "content": f"--- NEW {self.session_name.upper()} SESSION ---"
        }
        
        # Combine session-specific system prompt with messages
        isolated_messages = [boundary_message] + messages
        
        payload = {
            "model": self.model_config["model_name"],
            "messages": isolated_messages,
            "conversation_id": self.conversation_id,  # Unique per session
            **self.model_config["parameters"]  # Session-specific parameters
        }
        
        return payload
    
    def manage_conversation_memory(self, user_msg, bot_response):
        """Manage conversation history with size limits"""
        self.conversation_history.append((user_msg, bot_response))
        
        # Limit conversation history to prevent context overflow
        if len(self.conversation_history) > self.context_limit:
            self.conversation_history = self.conversation_history[-self.context_limit:]
    
    def reset_session(self):
        """Reset session state for clean start"""
        self.conversation_id = f"{self.session_name}_{int(time.time())}"
        self.conversation_history = []

# Emergency Medicine Session Manager
emergency_config = {
    "model_name": "deepseek/deepseek-r1-0528-qwen3-8b",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "stop": ["<|endoftext|>", "<|end|>", "---END_EMERGENCY---"],
        "max_tokens": 4096
    }
}

# Meeting Minutes Session Manager
meeting_config = {
    "model_name": "deepseek/deepseek-r1-0528-qwen3-8b", 
    "parameters": {
        "temperature": 0.3,
        "top_p": 0.95,
        "frequency_penalty": 0.05,
        "presence_penalty": 0.05,
        "stop": ["<|endoftext|>", "<|end|>", "---END_MEETING---"],
        "max_tokens": 4096
    }
}

# Create separate session managers
emergency_session = RAGSessionManager("emergency_medicine", emergency_config)
meeting_session = RAGSessionManager("meeting_minutes", meeting_config)
