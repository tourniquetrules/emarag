"""
Best practices for running multiple RAG systems on same LM Studio instance.
Implementation guidelines and code examples.
"""

# 1. SEPARATE PORTS (Current approach - works well)
# Emergency Medicine: http://localhost:7866  
# Meeting Minutes: http://localhost:7870

# 2. ENHANCED API CALL WITH SESSION ISOLATION
def call_lm_studio_api_isolated(messages, session_type="emergency", temperature=0.1, max_tokens=4096, stream=False):
    """Enhanced API call with session isolation parameters"""
    
    url = "http://10.5.0.2:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Session-specific configurations
    session_configs = {
        "emergency": {
            "temperature": 0.1,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stop": ["<|endoftext|>", "<|end|>", "---EMERGENCY_END---"],
            "session_prefix": "EMERGENCY_MED: "
        },
        "meeting": {
            "temperature": 0.3,
            "frequency_penalty": 0.05, 
            "presence_penalty": 0.05,
            "stop": ["<|endoftext|>", "<|end|>", "---MEETING_END---"],
            "session_prefix": "MEETING_MIN: "
        }
    }
    
    config = session_configs.get(session_type, session_configs["emergency"])
    
    # Add session prefix to first system message
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = config["session_prefix"] + messages[0]["content"]
    
    payload = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b",
        "messages": messages,
        "temperature": config["temperature"],
        "max_tokens": max_tokens,
        "stream": stream,
        "stop": config["stop"],
        "frequency_penalty": config["frequency_penalty"],
        "presence_penalty": config["presence_penalty"],
        # Additional isolation
        "conversation_id": f"{session_type}_{int(time.time())}",
        "reset_conversation": True  # Clear previous context
    }
    
    return requests.post(url, headers=headers, json=payload, timeout=120, stream=stream)

# 3. PERIODIC SESSION RESET
import threading
import time

def periodic_session_reset(interval_minutes=30):
    """Periodically reset LM Studio sessions to prevent long-term contamination"""
    
    def reset_sessions():
        while True:
            time.sleep(interval_minutes * 60)  # Convert to seconds
            
            # Send reset signal to both chatbots
            try:
                # Reset emergency medicine session
                reset_payload = {
                    "model": "deepseek/deepseek-r1-0528-qwen3-8b",
                    "messages": [{"role": "system", "content": "RESET_SESSION"}],
                    "reset_conversation": True,
                    "conversation_id": "emergency_reset"
                }
                requests.post("http://10.5.0.2:1234/v1/chat/completions", 
                            json=reset_payload, timeout=10)
                
                # Reset meeting minutes session  
                reset_payload["conversation_id"] = "meeting_reset"
                requests.post("http://10.5.0.2:1234/v1/chat/completions",
                            json=reset_payload, timeout=10)
                            
                print(f"🔄 Sessions reset at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                print(f"⚠️  Session reset failed: {e}")
    
    # Run in background thread
    reset_thread = threading.Thread(target=reset_sessions, daemon=True)
    reset_thread.start()

# 4. HEALTH CHECK AND CONTAMINATION DETECTION
def check_session_health():
    """Check both RAG systems for cross-contamination"""
    
    test_queries = {
        "emergency": "What is the first-line treatment for anaphylaxis?",
        "meeting": "What were the main discussion points in the last meeting?"
    }
    
    contamination_keywords = {
        "emergency": ["meeting", "agenda", "committee", "minutes"],
        "meeting": ["medication", "protocol", "diagnosis", "treatment"]
    }
    
    for session_type, query in test_queries.items():
        try:
            # Test query
            messages = [
                {"role": "system", "content": f"You are a {session_type} assistant."},
                {"role": "user", "content": query}
            ]
            
            response = call_lm_studio_api_isolated(messages, session_type)
            response_text = response.json()["choices"][0]["message"]["content"].lower()
            
            # Check for contamination
            bad_keywords = contamination_keywords[session_type]
            contaminated = any(keyword in response_text for keyword in bad_keywords)
            
            if contaminated:
                print(f"⚠️  CONTAMINATION DETECTED in {session_type} session!")
                print(f"   Response: {response_text[:100]}...")
                return False
            else:
                print(f"✅ {session_type} session healthy")
                
        except Exception as e:
            print(f"❌ Health check failed for {session_type}: {e}")
            return False
    
    return True

# 5. STARTUP CONFIGURATION
def initialize_multi_rag_system():
    """Initialize both RAG systems with proper isolation"""
    
    print("🚀 Initializing Multi-RAG System...")
    
    # Start periodic session reset
    periodic_session_reset(interval_minutes=30)
    
    # Initial health check
    if check_session_health():
        print("✅ Multi-RAG system initialized successfully")
    else:
        print("⚠️  Initial contamination detected - may need LM Studio restart")
    
    return True

# Example integration:
if __name__ == "__main__":
    initialize_multi_rag_system()
    
    # Your existing chatbot startup code here...
    print("Both RAG systems can now run safely on the same LM Studio instance!")
