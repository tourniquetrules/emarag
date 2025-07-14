"""
Context boundary implementation for preventing RAG contamination.
Uses clear markers to separate different knowledge domains.
"""

def create_emergency_medicine_prompt(query, relevant_chunks):
    """Create emergency medicine prompt with clear boundaries"""
    
    context_header = """
=== EMERGENCY MEDICINE KNOWLEDGE BASE ===
DOMAIN: Emergency Medicine, Clinical Protocols, Medical Research
CONTEXT TYPE: Medical Literature and Research Abstracts
SESSION: Emergency Medicine RAG System
===
"""
    
    if not relevant_chunks:
        return f"""{context_header}

You are an emergency medicine expert. Answer the following question based on your medical knowledge:

MEDICAL QUERY: {query}

Provide evidence-based emergency medicine guidance.

=== END EMERGENCY MEDICINE CONTEXT ==="""

    context = "\n\n".join([
        f"Source: {chunk['source']}\nContent: {chunk['text']}"
        for chunk in relevant_chunks
    ])
    
    prompt = f"""{context_header}

MEDICAL LITERATURE CONTEXT:
{context}

MEDICAL QUERY: {query}

Provide a comprehensive emergency medicine answer that:
1. References the provided medical literature
2. Focuses on emergency medicine practices and protocols  
3. Includes evidence-based recommendations
4. Considers safety and time-sensitive aspects of emergency care

=== END EMERGENCY MEDICINE CONTEXT ==="""

    return prompt

def create_meeting_minutes_prompt(query, relevant_chunks):
    """Create meeting minutes prompt with clear boundaries"""
    
    context_header = """
=== MEETING MINUTES KNOWLEDGE BASE ===
DOMAIN: Emergency Physicians Group Meetings
CONTEXT TYPE: Meeting Minutes, Discussions, Decisions
SESSION: Meeting Minutes RAG System
===
"""
    
    if not relevant_chunks:
        return f"""{context_header}

You are an assistant helping with emergency physicians group meeting information.

MEETING QUERY: {query}

Provide information based on meeting minutes and group discussions.

=== END MEETING MINUTES CONTEXT ==="""

    context = "\n\n".join([
        f"Meeting: {chunk['source']}\nContent: {chunk['text']}"
        for chunk in relevant_chunks
    ])
    
    prompt = f"""{context_header}

MEETING MINUTES CONTEXT:
{context}

MEETING QUERY: {query}

Provide information about:
1. Meeting discussions and decisions
2. Action items and assignments
3. Group policies and procedures
4. Administrative matters

=== END MEETING MINUTES CONTEXT ==="""

    return prompt

def validate_response_context(response, expected_context):
    """Validate that response stays within expected context boundaries"""
    
    contamination_indicators = {
        "emergency_medicine": [
            "meeting agenda", "action item", "committee decision", 
            "group policy", "administrative", "minutes show"
        ],
        "meeting_minutes": [
            "medical protocol", "clinical guideline", "emergency procedure",
            "patient care", "diagnosis", "treatment", "medication"
        ]
    }
    
    indicators = contamination_indicators.get(expected_context, [])
    response_lower = response.lower()
    
    contaminated = any(indicator in response_lower for indicator in indicators)
    
    if contaminated:
        print(f"⚠️  Potential context contamination detected in {expected_context} response")
        return False
    
    return True
