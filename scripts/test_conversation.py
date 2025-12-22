"""Test the conversation agent with specific scenarios."""
import sys
sys.path.insert(0, 'src')

from voice_platform.pipeline.llm_ollama import OllamaLLM
from voice_platform.conversation import ConversationAgent

# Initialize
print("Loading LLM...")
llm = OllamaLLM(model="llama3.2:latest", temperature=0.3)
llm.load()

print("\n" + "="*60)
print("TEST 1: User works mornings - should only get afternoon slots")
print("="*60)

agent = ConversationAgent(llm, clinic_name="Sunrise Dental")
response = agent.start()
print(f"\nBot: {response.message}")

# Test the morning availability constraint
test_inputs = [
    "Hi, I have really bad tooth pain and need an appointment",
    "I work mornings so only afternoons work, maybe Thursday?",
    "The 3pm one",  # Should pick the afternoon slot
    "Yes that's correct",
]

for user_input in test_inputs:
    print(f"\nUser: {user_input}")
    response = agent.process(user_input)
    print(f"Bot: {response.message}")
    print(f"Slots: {response.slots}")
    
    if response.ended:
        break

print("\n" + "="*60)
print("TEST 2: Ambiguous 'yes' after two options")  
print("="*60)

agent2 = ConversationAgent(llm, clinic_name="Sunrise Dental")
response = agent2.start()
print(f"\nBot: {response.message}")

test_inputs_2 = [
    "I need a cleaning appointment",
    "Thursday works for me",
    "yes",  # Ambiguous - should ask which time
]

for user_input in test_inputs_2:
    print(f"\nUser: {user_input}")
    response = agent2.process(user_input)
    print(f"Bot: {response.message}")
    print(f"Slots: {response.slots}")
    
    if response.ended:
        break

print("\n" + "="*60)
print("Tests complete!")
print("="*60)
