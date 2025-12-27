# Voice AI Platform - Architecture Document

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Config-Driven Design](#config-driven-design)
5. [Intent Classification System](#intent-classification-system)
6. [State Management](#state-management)
7. [Error Handling Strategy](#error-handling-strategy)
8. [Security & Compliance](#security--compliance)
9. [Design Decisions](#design-decisions)
10. [Performance Optimizations](#performance-optimizations)

---

## System Overview

### Purpose
Production-grade voice AI platform for healthcare appointment scheduling, demonstrating enterprise patterns suitable for FAANG/Big Tech interviews.

### Core Capabilities
- Real-time voice conversations (< 2s latency)
- Natural language understanding with context
- Config-driven conversation flows
- HIPAA-compliant logging and audit trails
- Multi-agent architecture for different use cases

### Technology Stack
| Layer | Technology | Purpose |
|-------|------------|---------|
| Voice Activity Detection | Silero VAD | Detect speech in audio stream |
| Speech-to-Text | Faster-Whisper | Convert audio to text |
| Language Model | Ollama (Llama 3.2) | Intent classification, response generation |
| Text-to-Speech | Kokoro / Piper | Convert text to natural speech |
| API Framework | FastAPI | REST/WebSocket endpoints |
| Configuration | YAML + Pydantic | Type-safe config management |
| Storage | SQLite / PostgreSQL | Appointment persistence |
| Caching | In-memory / Redis | Response caching |

---

## Component Architecture

### Module Dependency Graph
```
                    ┌─────────────────┐
                    │      core       │
                    │ config, types,  │
                    │ exceptions      │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │ logging │        │  audio  │        │ storage │
    └────┬────┘        └────┬────┘        └────┬────┘
         │                  │                   │
         │    ┌─────────────┼─────────────┐    │
         │    │             │             │    │
         ▼    ▼             ▼             ▼    ▼
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │   vad   │        │   asr   │        │   tts   │
    │ (Silero)│        │(Whisper)│        │(Kokoro) │
    └────┬────┘        └────┬────┘        └────┬────┘
         │                  │                   │
         └──────────────────┼───────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │      llm      │
                    │   (Ollama)    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    agent      │◀──── configs/agent/*.yaml
                    │ IntentClassif │
                    │ ToolCalling   │
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Inbound  │ │ Outbound │ │  Payer   │
        │  Agent   │ │  Agent   │ │  Agent   │
        └──────────┘ └──────────┘ └──────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   channels    │
                    │ WebRTC, WS,   │
                    │ Twilio        │
                    └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     api       │
                    │   FastAPI     │
                    └───────────────┘
```

### Module Responsibilities

| Module | Files | Purpose |
|--------|-------|---------|
| `core/` | config.py, types.py, exceptions.py, registry.py | Foundation layer |
| `logging/` | logger.py, audit.py | Structured logging, HIPAA audit |
| `audio/` | input.py | Audio I/O processing |
| `vad/` | silero.py, base.py | Voice Activity Detection |
| `asr/` | whisper.py, base.py | Speech-to-Text |
| `llm/` | ollama.py, streaming.py, base.py | LLM integration |
| `tts/` | kokoro.py, piper.py, base.py | Text-to-Speech |
| `agent/` | tool_calling_agent.py, extractors/* | Conversation intelligence |
| `channels/` | webrtc.py, websocket.py, twilio.py | Communication |
| `storage/` | database.py, redis.py | Persistence |

---

## Data Flow

### Voice Conversation Flow
```
User Speaks
    │
    ▼
┌─────────────────────────────────────┐
│ 1. AUDIO CAPTURE (~50ms)            │
│    WebRTC / WebSocket / Microphone  │
│    16kHz, 16-bit mono               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. VAD - Silero (~20ms)             │
│    Detect speech vs silence         │
│    Buffer speech segments           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. ASR - Faster-Whisper (~300ms)    │
│    Transcribe audio to text         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. INTENT CLASSIFICATION            │
│    Fast Path: ~0ms (regex)          │
│    LLM Path: ~500ms (Ollama)        │
│    Returns: intent, action, value   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. ACTION EXECUTION                 │
│    Update state, extract values     │
│    Generate response                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. TTS - Kokoro (~200ms)            │
│    Convert text to speech           │
└─────────────────────────────────────┘
    │
    ▼
User Hears Response
```

---

## Config-Driven Design

### Philosophy
**"Behavior in YAML, mechanics in Python"**

### Config Files
```
configs/agent/
├── states.yaml     # State definitions & transitions
├── actions.yaml    # Action behaviors
├── examples.yaml   # Few-shot examples for LLM
└── prompts.yaml    # System & user prompts
```

### State Configuration
```yaml
# configs/agent/states.yaml
states:
  collecting_name:
    question: "May I have your full name please?"
    required: true
    field: name
    implicit_confirmation: "Thanks {value}."
    transitions:
      on_accept: collecting_dob
```

### Action Configuration
```yaml
# configs/agent/actions.yaml
actions:
  accept_input:
    behavior:
      - pass_through: true  # Let state handler process
      
  end_call:
    behavior:
      - set_state: complete
      - use_suggested_response: true
      - fallback: "Goodbye!"
```

### Benefits
- No code changes for flow updates
- A/B testing via config swaps
- Domain adaptation (dental/medical/veterinary)
- Git-tracked audit trail

---

## Intent Classification System

### Hybrid Architecture
```
User Input
    │
    ├──▶ Fast Path (Regex) ──▶ Result (~0ms)
    │         │
    │         │ Ambiguous
    │         ▼
    └──▶ LLM Path (Ollama) ──▶ Result (~500ms)
```

### Classification Result
```python
@dataclass
class ClassificationResult:
    intent: UserIntent       # providing_info, refusing, correcting, etc.
    action: Action           # accept_input, ask_again, end_call, etc.
    extracted_value: str     # "John Smith"
    field_to_correct: str    # "name"
    suggested_response: str  # "Thanks John..."
    confidence: float        # 0.95
```

### Context-Aware Actions
| User Says | State | Intent | Action |
|-----------|-------|--------|--------|
| "No" | consent | refusing | end_call |
| "No" | name | refusing | ask_again |
| "No" | confirming | denying | ask_clarification |

---

## State Management

### State Flow
```
COLLECTING_CONSENT → COLLECTING_NAME → COLLECTING_DOB
         │                                    │
         ▼                                    ▼
      end_call                         COLLECTING_PHONE
                                              │
                                              ▼
                                       COLLECTING_REASON
                                              │
                                              ▼
                                       COLLECTING_DAY
                                              │
                                              ▼
                                       COLLECTING_TIME
                                              │
                                              ▼
                                         CONFIRMING
                                              │
                                              ▼
                                          COMPLETE
```

### Correction Flow
User can correct from any state:
```
State: collecting_dob
User: "My name is Jane, not John"
  │
  ▼
Action: correct_field
Field: name
Value: "Jane"
  │
  ▼
Update name, continue from collecting_dob
```

---

## Error Handling Strategy

### Exception Hierarchy
```
VoicePlatformError
├── ConfigError
├── AudioError
│   ├── VADError
│   └── ASRError
├── LLMError
│   ├── LLMConnectionError
│   └── LLMTimeoutError
├── TTSError
└── StorageError
```

### Retry with Backoff
```python
for attempt in range(max_retries):
    try:
        return self._llm_classify(text, context)
    except Exception:
        time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
return self._fallback_result()
```

---

## Security & Compliance

### PHI Redaction
```python
# Logs show: "Joh***", "555-***-4567", "**/**/****"
def _mask_phi(self, name, phone, dob):
    return {
        "name": name[:3] + "***",
        "phone": phone[:3] + "-***-" + phone[-4:],
        "dob": "**/**/****"
    }
```

### HIPAA Audit Trail
- Session-based isolation
- All data access logged
- No PHI in application logs
- Hashed patient IDs

---

## Design Decisions

| Decision | Chosen | Rationale |
|----------|--------|-----------|
| Config vs Hardcoded | Config-driven | Enterprise flexibility |
| Intent only vs Intent+Action | Intent+Action | Context-aware responses |
| LLM only vs Hybrid | Hybrid | Latency optimization |
| Explicit vs Implicit confirmation | Implicit | Natural conversation |

---

## Performance Optimizations

| Optimization | Impact |
|--------------|--------|
| Response caching | Repeated inputs: ~0ms |
| Fast path regex | 20% of inputs: ~0ms |
| Example selection | Fewer tokens, faster LLM |
| Lazy initialization | Faster startup |

### Latency Budget
```
Audio capture:     ~50ms
VAD:               ~20ms
ASR:              ~300ms
Intent classify:  ~500ms (LLM) / ~0ms (fast)
LLM response:     ~400ms
TTS:              ~200ms
─────────────────────────
Total:            ~1.5s
```
