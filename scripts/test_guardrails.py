"""Comprehensive guardrail and edge case tests."""
import sys
sys.path.insert(0, 'src')

from voice_platform.pipeline.llm_ollama import OllamaLLM
from voice_platform.conversation import ConversationAgent

def run_test(name: str, inputs: list, expected_behavior: str):
    """Run a single test scenario."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Expected: {expected_behavior}")
    print('='*60)
    
    agent = ConversationAgent(llm, clinic_name="Sunrise Dental")
    response = agent.start()
    print(f"\nBot: {response.message}")
    
    for user_input in inputs:
        print(f"\nUser: {user_input}")
        response = agent.process(user_input)
        print(f"Bot: {response.message}")
        print(f"Slots: {response.slots}")
        print(f"Stage: {response.stage.value}")
        
        if response.ended:
            break
    
    return response

# Initialize LLM once
print("Loading LLM...")
llm = OllamaLLM(model="llama3.2:latest", temperature=0.3)
llm.load()

print("\n" + "#"*60)
print("# GUARDRAIL TESTS")
print("#"*60)

# ============================================================
# 1. EMERGENCY DETECTION
# ============================================================
run_test(
    "Emergency - Severe Pain",
    [
        "I'm having severe chest pain and difficulty breathing",
    ],
    "Should recognize medical emergency and suggest calling 911 or transfer"
)

run_test(
    "Emergency - Dental Emergency",
    [
        "I fell and knocked out my front tooth, there's blood everywhere",
    ],
    "Should prioritize getting them in ASAP or direct to emergency care"
)

# ============================================================
# 2. TRANSFER REQUESTS
# ============================================================
run_test(
    "Explicit Transfer Request",
    [
        "Can I speak to a real person please?",
    ],
    "Should immediately offer to transfer"
)

run_test(
    "Frustrated User Wants Human",
    [
        "I need an appointment",
        "Thursday",
        "This is ridiculous, just let me talk to someone",
    ],
    "Should transfer when user expresses frustration with AI"
)

# ============================================================
# 3. OFF-TOPIC HANDLING
# ============================================================
run_test(
    "Completely Off Topic",
    [
        "What's the weather like today?",
    ],
    "Should redirect to scheduling or offer transfer"
)

run_test(
    "Wrong Business",
    [
        "I'd like to order a pizza please",
    ],
    "Should politely clarify this is a dental office"
)

# ============================================================
# 4. SLOT MODIFICATION
# ============================================================
run_test(
    "Change Day After Stating",
    [
        "I need a cleaning",
        "Thursday works",
        "Actually wait, make that Tuesday instead",
    ],
    "Should update the day to Tuesday"
)

run_test(
    "Change Everything",
    [
        "Tooth pain, Thursday 3pm",
        "Actually I need to completely change that - it's for a cleaning on Monday morning",
    ],
    "Should update all slots to new values"
)

# ============================================================
# 5. CLARIFICATION REQUESTS
# ============================================================
run_test(
    "User Asks to Repeat",
    [
        "I need a checkup",
        "Tuesday",
        "Sorry, what times did you say were available?",
    ],
    "Should repeat the available time options"
)

run_test(
    "User Confused",
    [
        "I need an appointment",
        "Wait, is this the dentist or the doctor?",
    ],
    "Should clarify this is a dental office"
)

# ============================================================
# 6. FAQ HANDLING
# ============================================================
run_test(
    "Insurance Question",
    [
        "Do you accept Blue Cross Blue Shield?",
    ],
    "Should handle insurance question, then redirect to scheduling"
)

run_test(
    "Location Question",
    [
        "What's your address?",
    ],
    "Should provide info or offer to help, then redirect to scheduling"
)

run_test(
    "Hours Question",
    [
        "What are your hours?",
    ],
    "Should provide hours info"
)

# ============================================================
# 7. GIBBERISH/NOISE
# ============================================================
run_test(
    "Unclear Speech",
    [
        "asdkjfh aksjdfh kajsdhf",
    ],
    "Should ask user to repeat"
)

run_test(
    "Very Short Response",
    [
        "um",
    ],
    "Should prompt for more information"
)

# ============================================================
# 8. MULTI-INTENT
# ============================================================
run_test(
    "Multiple Requests",
    [
        "I need to schedule a cleaning and also ask about my bill",
    ],
    "Should handle scheduling, offer to transfer for billing"
)

# ============================================================
# 9. CANCELLATION (Different Flow)
# ============================================================
run_test(
    "Cancel Existing Appointment",
    [
        "I need to cancel my appointment for tomorrow",
    ],
    "Should handle cancellation or transfer to staff"
)

# ============================================================
# 10. CONVERSATION LIMITS
# ============================================================
run_test(
    "Long Conversation Without Progress",
    [
        "hi",
        "yeah",
        "ok",
        "sure",
        "maybe",
        "I don't know",
        "whatever",
    ],
    "Should try to guide conversation or offer transfer after too many non-productive turns"
)

print("\n" + "#"*60)
print("# TESTS COMPLETE")
print("#"*60)
print("\nReview the outputs above to identify which guardrails need implementation.")
