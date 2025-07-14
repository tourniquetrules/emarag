import gradio as gr
import requests
import json
import time
import re

# LM Studio API configuration
LM_STUDIO_BASE_URL = "http://10.5.0.2:1234"
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"

# Global variables for chat history and metrics
total_tokens_generated = 0
total_time_spent = 0
total_requests = 0
conversation_id = None
max_conversation_length = 50  # Maximum number of message pairs to keep in context

def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 4 characters for most models)"""
    return len(text) // 4

def process_deepseek_response(response_text):
    """Process DeepSeek response to extract the final answer from reasoning"""
    if not response_text:
        return "No response received."
    
    # DeepSeek uses <think> tags for reasoning, extract the final answer
    if "<think>" in response_text and "</think>" in response_text:
        # Split on </think> and get everything after it
        parts = response_text.split("</think>")
        if len(parts) > 1:
            final_answer = parts[-1].strip()
            if final_answer:
                return final_answer
        
        # If no content after </think>, try to extract from inside think tags
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, response_text, re.DOTALL)
        if think_matches:
            # Look for a clear answer in the reasoning
            reasoning = think_matches[-1]
            # Try to find a clear answer pattern
            answer_patterns = [
                r'[Tt]he answer (?:is|should be|would be) (.+?)(?:\.|$)',
                r'[Ss]o (?:the answer is|it\'s) (.+?)(?:\.|$)',
                r'[Tt]herefore,? (.+?)(?:\.|$)',
                r'[Tt]hus,? (.+?)(?:\.|$)',
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, reasoning)
                if match:
                    return match.group(1).strip()
            
            # If no pattern matches, return the last sentence of reasoning
            sentences = reasoning.strip().split('.')
            if sentences and sentences[-1].strip():
                return sentences[-1].strip()
    
    # If no think tags, return the response as-is
    return response_text.strip()

def call_lm_studio_api(messages, temperature=0.7, max_tokens=16384, stream=False):
    """Call LM Studio API with the given messages"""
    url = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    try:
        print(f"🔄 Calling LM Studio API with {len(messages)} messages (streaming: {stream})")
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=120, stream=stream)
        response.raise_for_status()
        
        if stream:
            # Handle streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    full_response += delta['content']
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            raw_response = full_response
            # For streaming, we estimate usage since it's not always provided
            usage = {"completion_tokens": estimate_tokens(raw_response)}
        else:
            # Handle non-streaming response
            end_time = time.time()
            data = response.json()
            raw_response = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
        
        # Process the response to extract the final answer
        processed_response = process_deepseek_response(raw_response)
        
        # Calculate metrics
        response_time = end_time - start_time
        completion_tokens = usage.get("completion_tokens", estimate_tokens(raw_response))
        tokens_per_second = completion_tokens / response_time if response_time > 0 else 0
        
        print(f"✅ API Success: {completion_tokens} tokens in {response_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        print(f"📝 Raw response preview: {raw_response[:150]}...")
        print(f"🎯 Processed answer: {processed_response}")
        
        return {
            "success": True,
            "message": processed_response,
            "raw_message": raw_response,
            "response_time": response_time,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tokens_per_second
        }
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error: {e}"
        }

def check_api_health():
    """Check if LM Studio API is available"""
    try:
        response = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False

def chat_response(message, history, temperature, max_tokens, system_prompt, show_reasoning, enable_streaming, context_length):
    """Generate chat response using LM Studio API"""
    global total_tokens_generated, total_time_spent, total_requests, conversation_id, max_conversation_length
    
    # Update context length from UI
    max_conversation_length = int(context_length)
    
    if not message.strip():
        return history, "Please enter a message.", ""
    
    # Generate conversation ID if this is the first message
    if conversation_id is None:
        conversation_id = f"conv_{int(time.time())}"
        print(f"🆔 Started new conversation: {conversation_id}")
    
    # Check API health
    if not check_api_health():
        return history, f"❌ Cannot connect to LM Studio API at {LM_STUDIO_BASE_URL}", ""
    
    # Prepare messages for API with context management
    messages = []
    
    # Add system prompt if provided
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    
    # Manage conversation length to prevent context overflow
    context_history = history[-max_conversation_length:] if len(history) > max_conversation_length else history
    
    # Add chat history (maintaining full conversation context)
    for user_msg, bot_msg in context_history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    print(f"📤 Sending {len(messages)} messages to API (Context: {len(context_history)} pairs)")
    
    # Call API with streaming option
    result = call_lm_studio_api(messages, temperature, max_tokens, stream=enable_streaming)
    
    if result["success"]:
        # Update global metrics
        total_tokens_generated += result["completion_tokens"]
        total_time_spent += result["response_time"]
        total_requests += 1
        
        # Always show the processed (clean) answer in chat history
        # The reasoning display will show the full details if requested
        chat_display_message = result["message"]
        
        # Add to history - always use the clean answer for chat
        history.append([message, chat_display_message])
        
        # Generate metrics string
        metrics_str = (f"⏱️ {result['response_time']:.2f}s | "
                      f"🔢 {result['completion_tokens']} tokens | "
                      f"⚡ {result['tokens_per_second']:.1f} tok/s")
        
        # Generate reasoning display
        reasoning_display = ""
        if show_reasoning:
            if result["raw_message"] != result["message"]:
                reasoning_display = f"**Raw Response (with reasoning):**\n```\n{result['raw_message']}\n```\n\n**Processed Answer:**\n{result['message']}"
            else:
                reasoning_display = f"**Response:**\n{result['message']}"
        else:
            reasoning_display = f"**Clean Answer:** {result['message']}"
        
        return history, metrics_str, reasoning_display
    else:
        return history, f"❌ Error: {result['message']}", ""

def clear_chat():
    """Clear chat history and reset conversation state"""
    global total_tokens_generated, total_time_spent, total_requests, conversation_id
    total_tokens_generated = 0
    total_time_spent = 0
    total_requests = 0
    conversation_id = None
    print("🧹 Chat cleared, conversation state reset")
    return [], "Chat cleared and conversation state reset.", ""

def get_session_stats():
    """Get session statistics"""
    global total_tokens_generated, total_time_spent, total_requests, conversation_id
    
    if total_requests == 0:
        return "No messages sent yet."
    
    avg_tokens_per_second = total_tokens_generated / total_time_spent if total_time_spent > 0 else 0
    avg_response_time = total_time_spent / total_requests
    
    stats = f"""📊 **Session Statistics:**

**Conversation ID:** {conversation_id or 'No conversation started'}
**Total Requests:** {total_requests}
**Total Tokens Generated:** {total_tokens_generated:,}
**Total Time Spent:** {total_time_spent:.2f} seconds
**Average Response Time:** {avg_response_time:.2f} seconds
**Average Speed:** {avg_tokens_per_second:.1f} tokens/second
**Context Management:** Last {max_conversation_length} message pairs kept

**Model:** {MODEL_NAME}
**API Endpoint:** {LM_STUDIO_BASE_URL}
"""
    return stats

# Create the interface
with gr.Blocks(title="LM Studio Chat - DeepSeek Ready", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 💬 LM Studio Chat Interface - DeepSeek Ready
    
    Chat with your local LM Studio model. This version properly handles DeepSeek's reasoning format.
    
    **Model:** `deepseek/deepseek-r1-0528-qwen3-8b`
    **API:** `http://127.0.0.1:1234`
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Main chat interface
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
                show_label=True
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Metrics display
            metrics_display = gr.Textbox(
                label="Last Response Metrics",
                value="Send a message to see metrics...",
                interactive=False,
                lines=2
            )
            
            # Reasoning display
            reasoning_display = gr.Markdown(
                label="Response Details",
                value="Response details will appear here...",
            )
            
        with gr.Column(scale=1):
            # Settings panel
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")
                
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="You are a helpful assistant...",
                    lines=3,
                    value="You are a helpful, harmless, and honest assistant. Give clear, direct answers."
                )
                
                show_reasoning = gr.Checkbox(
                    label="Show Reasoning Process",
                    value=False,
                    info="Show the full DeepSeek reasoning or just the final answer"
                )
                
                enable_streaming = gr.Checkbox(
                    label="Enable Token Streaming",
                    value=True,
                    info="Stream tokens as they are generated (faster perceived response)"
                )
                
                context_length = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Conversation Context Length",
                    info="Number of message pairs to keep in context (prevents memory overflow)"
                )
                
                gr.Markdown("**Temperature:** Consistency (math, coding) ← → Creativity (writing)")
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                
                gr.Markdown("**Max Tokens:** Short concise answers ← → Detailed comprehensive responses")
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=16384,
                    value=4096,
                    step=128,
                    label="Max Tokens (Up to 16K)"
                )
            
            # Control buttons
            with gr.Group():
                gr.Markdown("### 🔧 Controls")
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                stats_btn = gr.Button("Show Stats", variant="secondary")
            
            # Stats display
            stats_display = gr.Textbox(
                label="Session Statistics",
                value="No statistics yet...",
                interactive=False,
                lines=10
            )
    
    # Event handlers
    send_btn.click(
        fn=chat_response,
        inputs=[msg_input, chatbot, temperature, max_tokens, system_prompt, show_reasoning, enable_streaming, context_length],
        outputs=[chatbot, metrics_display, reasoning_display]
    ).then(
        lambda: "",  # Clear the input
        outputs=msg_input
    )
    
    msg_input.submit(
        fn=chat_response,
        inputs=[msg_input, chatbot, temperature, max_tokens, system_prompt, show_reasoning, enable_streaming, context_length],
        outputs=[chatbot, metrics_display, reasoning_display]
    ).then(
        lambda: "",  # Clear the input
        outputs=msg_input
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, metrics_display, reasoning_display]
    )
    
    stats_btn.click(
        fn=get_session_stats,
        outputs=stats_display
    )

if __name__ == "__main__":
    print("🚀 Starting DeepSeek-Ready LM Studio Chat Interface...")
    print(f"📡 API Endpoint: {LM_STUDIO_BASE_URL}")
    print(f"🤖 Model: {MODEL_NAME}")
    
    # Check API health at startup
    if check_api_health():
        print("✅ LM Studio API is accessible")
    else:
        print("❌ Warning: Cannot connect to LM Studio API")
    
    print("\n" + "="*50)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,  # New port
        share=False,
        show_error=True
    )
