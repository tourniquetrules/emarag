"""
Example of session isolation for running multiple RAG systems on same LM Studio instance.
This prevents cross-contamination between emergency medicine and meeting minutes chatbots.
"""

# Emergency Medicine RAG Session Parameters
EMERGENCY_SESSION_CONFIG = {
    "session_id": "emergency_medicine",
    "temperature": 0.1,  # More factual for medical content
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
    "stop": ["<|endoftext|>", "<|end|>", "---END_MEDICAL---"],
    "system_prefix": "EMERGENCY_MED_SESSION: ",
    "conversation_memory": False  # Disable cross-session memory
}

# Meeting Minutes RAG Session Parameters  
MEETING_SESSION_CONFIG = {
    "session_id": "meeting_minutes",
    "temperature": 0.3,  # Slightly more creative for meeting summaries
    "top_p": 0.95,
    "frequency_penalty": 0.05,
    "presence_penalty": 0.05,
    "stop": ["<|endoftext|>", "<|end|>", "---END_MEETING---"],
    "system_prefix": "MEETING_MIN_SESSION: ",
    "conversation_memory": False  # Disable cross-session memory
}

def call_lm_studio_with_session_isolation(messages, session_config, model_name):
    """
    Call LM Studio API with session isolation parameters to prevent contamination.
    """
    import requests
    import json
    
    url = "http://10.5.0.2:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Add session prefix to system message to create clear context boundaries
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = session_config["system_prefix"] + messages[0]["content"]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": session_config["temperature"],
        "top_p": session_config["top_p"],
        "frequency_penalty": session_config["frequency_penalty"],
        "presence_penalty": session_config["presence_penalty"],
        "stop": session_config["stop"],
        "max_tokens": 4096,
        "stream": True,
        # Session isolation parameters
        "session_id": session_config["session_id"],  # If supported by LM Studio
        "conversation_memory": session_config["conversation_memory"]
    }
    
    return requests.post(url, headers=headers, json=payload, stream=True)

# Example usage in emergency medicine chatbot:
def emergency_medicine_call(messages):
    return call_lm_studio_with_session_isolation(
        messages, 
        EMERGENCY_SESSION_CONFIG, 
        "deepseek/deepseek-r1-0528-qwen3-8b"
    )

# Example usage in meeting minutes chatbot:
def meeting_minutes_call(messages):
    return call_lm_studio_with_session_isolation(
        messages, 
        MEETING_SESSION_CONFIG, 
        "deepseek/deepseek-r1-0528-qwen3-8b"
    )
