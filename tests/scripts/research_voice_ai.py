"""Research: Modern voice AI conversation approaches."""

# The problem with current approach:
# 1. State machine (YAML flow) controls conversation
# 2. LLM is just a "fallback" for intent matching
# 3. Flow asks questions even when answers are already known
# 4. Rigid, not natural

# Modern approaches:

# 1. RASA / Dialogflow (Traditional NLU)
#    - Intent classifier + NER for slots
#    - Forms for slot collection
#    - Stories/flows for paths
#    - Requires training data

# 2. LLM-Centric (Modern - what we should use)
#    - LLM is the BRAIN, not fallback
#    - Single call: understand + extract + decide + respond
#    - Structured output (JSON) for reliability
#    - Minimal state machine - just stages

# 3. Function Calling / Tool Use
#    - LLM decides when to call tools
#    - Calendar API, booking system, etc.
#    - Most flexible

# Best approach for our use case:
# - LLM generates response + extracts slots in ONE call
# - Simple stage tracking (not 15 states)
# - LLM decides what to ask based on missing slots
# - Confirmation before action

print("""
RECOMMENDED ARCHITECTURE:

┌─────────────────────────────────────────────────┐
│                   USER INPUT                     │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              LLM (Single Call)                   │
│  ┌───────────────────────────────────────────┐  │
│  │ System Prompt:                            │  │
│  │ - You are appointment scheduler           │  │
│  │ - Required slots: reason, day, time       │  │
│  │ - Current slots: {filled_slots}           │  │
│  │ - Stage: {current_stage}                  │  │
│  │                                           │  │
│  │ Output JSON:                              │  │
│  │ {                                         │  │
│  │   "slots": {...extracted...},             │  │
│  │   "response": "...",                      │  │
│  │   "ready_to_book": true/false,            │  │
│  │   "needs_clarification": null/"..."       │  │
│  │ }                                         │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              SIMPLE STAGES                       │
│  greeting → collecting → confirming → done      │
└─────────────────────────────────────────────────┘

This is how Retell AI, Vapi, and modern voice AI works.
""")
