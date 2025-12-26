"""Test LLM JSON output."""
import sys
sys.path.insert(0, 'src')

from voice_platform.pipeline.llm_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:latest", temperature=0.3)
llm.load()

system = """You are a medical appointment scheduler. Extract information and respond.

Respond with JSON only:
{
    "extracted_slots": {"visit_reason": "...", "preferred_day": "...", "preferred_time": "..."},
    "response": "your reply",
    "stage": "collecting"
}"""

prompt = """User said: "Hi, I have really bad tooth pain and need an appointment Thursday afternoon"

Respond with JSON:"""

print("Calling LLM...")
response = llm.generate(prompt, system)
print(f"Raw response:\n{response}")
print(f"\nLength: {len(response)}")
