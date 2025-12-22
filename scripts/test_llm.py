"""Test LLM integration."""
import sys
sys.path.insert(0, 'src')

from rich.console import Console
console = Console()

# Setup logging
from voice_platform.logging import setup_logging
from voice_platform.core.config import LoggingConfig
setup_logging(LoggingConfig(level="INFO", format="console"))

console.print("\n[bold cyan]Testing Ollama LLM[/]\n")

from voice_platform.pipeline.llm_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1:8b", temperature=0.1)
llm.load()

# Test 1: Intent classification
console.print("[yellow]Test 1: Intent Classification[/]")
intents = ["schedule", "cancel", "reschedule", "hours"]

test_inputs = [
    "I need to book an appointment",
    "Can you cancel my visit?",
    "I want to move my appointment to next week",
    "What time do you close?",
    "My tooth is killing me, can I come in Thursday?",
]

for text in test_inputs:
    intent = llm.classify_intent(text, intents, context="Medical clinic appointment system")
    console.print(f"  '{text}' → [green]{intent}[/]")

# Test 2: Slot extraction
console.print("\n[yellow]Test 2: Slot Extraction[/]")
slots = {
    "preferred_day": "day of the week",
    "preferred_time": "time of day (morning/afternoon/evening or specific time)",
    "visit_reason": "reason for visit",
}

test_inputs = [
    "I'd like to come in Thursday morning for a checkup",
    "Can I get an appointment next Monday at 2pm?",
    "My back has been hurting, maybe Wednesday afternoon?",
    "anytime Tuesday works for me",
]

for text in test_inputs:
    extracted = llm.extract_slots(text, slots, context="Scheduling appointment")
    console.print(f"  '{text}'")
    console.print(f"    → [green]{extracted}[/]")

console.print("\n[bold green]LLM tests complete![/]")
