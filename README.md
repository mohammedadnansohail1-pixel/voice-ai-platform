# Voice AI Platform

A production-grade, config-driven voice AI platform for healthcare appointment scheduling. Built with enterprise patterns, HIPAA-compliant logging, and multi-agent architecture.

## 🎯 Project Overview

This platform enables natural voice conversations for booking medical appointments. It demonstrates:

- **Real-time voice processing** with sub-second latency
- **Config-driven architecture** - behavior changes via YAML, not code
- **Production patterns** - error handling, logging, metrics, caching
- **Multi-agent design** - Inbound, Outbound, and Payer agents

### Key Differentiators

| Feature | Traditional Approach | This Platform |
|---------|---------------------|---------------|
| Intent Classification | Hardcoded if/else | Config-driven + LLM |
| Conversation Flow | State machine in code | YAML-defined states |
| Error Handling | Ad-hoc | Comprehensive exception hierarchy |
| Logging | Basic print | Structured logging with PHI redaction |

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        Voice AI Platform                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │   VAD   │───▶│   ASR   │───▶│   LLM   │───▶│   TTS   │       │
│  │ Silero  │    │ Whisper │    │ Ollama  │    │ Kokoro  │       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│       │              │              │              │             │
│       └──────────────┴──────────────┴──────────────┘             │
│                            │                                      │
│                    ┌───────▼───────┐                             │
│                    │ Intent        │                             │
│                    │ Classifier    │◀── configs/agent/*.yaml     │
│                    │ (Config-Driven)│                            │
│                    └───────┬───────┘                             │
│                            │                                      │
│                    ┌───────▼───────┐                             │
│                    │ Tool Calling  │                             │
│                    │ Agent         │                             │
│                    └───────┬───────┘                             │
│                            │                                      │
│         ┌──────────────────┼──────────────────┐                  │
│         │                  │                  │                  │
│  ┌──────▼──────┐   ┌───────▼──────┐   ┌──────▼──────┐          │
│  │   Inbound   │   │   Outbound   │   │    Payer    │          │
│  │   Agent     │   │    Agent     │   │    Agent    │          │
│  │ (Scheduling)│   │  (Reminders) │   │ (Insurance) │          │
│  └─────────────┘   └──────────────┘   └─────────────┘          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure (12 Focused Modules)
```
src/voice_platform/
├── core/           # Foundation: config, types, exceptions, registry
├── logging/        # Structured logging with HIPAA audit trails
├── audio/          # Audio I/O processing
├── vad/            # Voice Activity Detection (Silero)
├── asr/            # Speech-to-Text (Faster-Whisper)
├── llm/            # LLM integration (Ollama, streaming)
├── tts/            # Text-to-Speech (Kokoro, Piper)
├── agent/          # Conversation agents and extractors
├── channels/       # Communication channels (WebRTC, WebSocket, Twilio)
├── storage/        # Persistence (Redis, PostgreSQL, SQLite)
├── pipelines/      # End-to-end processing pipelines
└── api/            # FastAPI endpoints
```

---

## 🎛️ Config-Driven Design

All conversation behavior is defined in YAML configs, not code:

### States (`configs/agent/states.yaml`)
```yaml
states:
  collecting_name:
    question: "May I have your full name please?"
    required: true
    field: name
    implicit_confirmation: "Thanks {value}. If I got that wrong, just let me know."
    transitions:
      on_accept: collecting_dob
```

### Actions (`configs/agent/actions.yaml`)
```yaml
actions:
  accept_input:
    description: "Accept the input and proceed"
    behavior:
      - use_extracted_value: true
      - transition: on_accept
      
  end_call:
    description: "End the conversation gracefully"
    behavior:
      - set_state: complete
      - use_suggested_response: true
      - fallback: "Thank you for calling. Goodbye!"
```

### Few-Shot Examples (`configs/agent/examples.yaml`)
```yaml
examples:
  - context:
      assistant: "May I have your full name please?"
      state: collecting_name
    user: "John Smith"
    classification:
      intent: providing_info
      action: accept_input
      extracted_value: "John Smith"
      confidence: 0.95
```

### Benefits
- **No code changes** for conversation flow updates
- **A/B testing** - swap config files
- **Domain adaptation** - same code, different configs for dental/medical/veterinary
- **Audit trail** - config changes are git-tracked

---

## 🧠 Intent Classification

Hybrid approach combining fast regex patterns with LLM fallback:
```
User Input
    │
    ▼
┌───────────────┐
│  Fast Path    │──── "Yes" at consent ──▶ CONTINUE (0ms)
│  (Regex)      │──── "No" at consent ───▶ END_CALL (0ms)
└───────┬───────┘
        │ Ambiguous
        ▼
┌───────────────┐
│  LLM Path     │──── Few-shot examples
│  (Ollama)     │──── Full context
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Classification │
│ Result         │
│ - intent       │
│ - action       │
│ - value        │
│ - response     │
└────────────────┘
```

### Intents
| Intent | Description | Example |
|--------|-------------|---------|
| `providing_info` | User gives requested data | "John Smith" |
| `confirming` | User agrees | "Yes", "Correct" |
| `refusing` | User declines | "No", "I don't want to" |
| `denying` | User says something is wrong | "That's incorrect" |
| `correcting` | User provides correction | "My name is Jane, not John" |
| `off_topic` | Unrelated to current question | "Hello", "How are you" |

### Actions
| Action | Behavior |
|--------|----------|
| `accept_input` | Let state handler process input |
| `continue` | Proceed to next state |
| `ask_again` | Re-ask current question |
| `ask_clarification` | Ask what user wants to fix |
| `correct_field` | Update specific field |
| `end_call` | Gracefully end conversation |

---

## 🔧 Technical Decisions

### 1. Config-Driven vs Hardcoded
**Decision:** All conversation logic in YAML
**Rationale:** 
- Enables non-developers to modify behavior
- Facilitates A/B testing
- Separates logic from implementation
- Aligns with enterprise deployment patterns

### 2. Hybrid Intent Classification
**Decision:** Fast regex path + LLM fallback
**Rationale:**
- Fast path handles 80% of clear inputs (0ms latency)
- LLM handles ambiguous cases with full context
- Industry standard (Retell AI, Vapi, Alexa)

### 3. Action-Based Response
**Decision:** LLM returns `action` not just `intent`
**Rationale:**
- "No" means different things based on context
- Same intent (`refusing`) → different actions (`ask_again` vs `end_call`)
- Removes hardcoded if/else in agent

### 4. Implicit Confirmation
**Decision:** "Thanks John. If I got that wrong, just say so."
**Rationale:**
- Reduces conversation turns
- More natural flow
- User can correct inline without explicit confirmation step

### 5. Pydantic Validation
**Decision:** All configs validated with Pydantic models
**Rationale:**
- Fail fast on invalid config
- Type safety
- Self-documenting schemas

---

## 📊 Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| End-to-end latency | < 2s | ~1.5s |
| Fast path classification | < 10ms | ~0ms |
| LLM classification | < 1s | ~500-800ms |
| VAD accuracy | > 95% | ~97% (Silero) |

### Latency Breakdown
```
Audio capture:     ~50ms
VAD detection:     ~20ms
ASR (Whisper):     ~300ms
Intent classify:   ~500ms (LLM) / ~0ms (fast path)
LLM response:      ~400ms
TTS synthesis:     ~200ms
─────────────────────────
Total:             ~1.5s
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Ollama with `llama3.2:latest`
- 16GB+ RAM recommended

### Installation
```bash
# Clone repository
git clone https://github.com/mohammedadnansohail1-pixel/voice-ai-platform.git
cd voice-ai-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Start Ollama
ollama pull llama3.2:latest
ollama serve
```

### Run Demo
```bash
# WebRTC voice demo
python scripts/run_webrtc_demo.py

# Text mode testing
python scripts/test_text_mode.py
```

### Configuration
```bash
# Modify conversation flow
vim configs/agent/states.yaml

# Add few-shot examples
vim configs/agent/examples.yaml

# Update prompts
vim configs/agent/prompts.yaml
```

---

## 🧪 Testing
```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Test intent classifier
python -c "
from voice_platform.agent.extractors import IntentClassifier
c = IntentClassifier()
result = c.classify('John Smith', {'state': 'collecting_name', 'last_assistant_message': 'May I have your name?', 'collected': {}})
print(f'{result.intent.value} / {result.action.value}')
"
```

---

## 📁 Project History

### Phase 1: Foundation
- Core voice pipeline (VAD → ASR → LLM → TTS)
- WebSocket/WebRTC channels
- Basic slot extraction

### Phase 2: Production Hardening
- Comprehensive error handling
- Structured logging with PHI redaction
- HIPAA-compliant audit trails

### Phase 3: Config-Driven Architecture
- YAML-based state definitions
- Few-shot example configs
- Action-based intent classification
- Removed all hardcoded if/else logic

### Phase 4: Multi-Agent (Current)
- Inbound Agent (appointment scheduling) ✅
- Outbound Agent (reminders) 🔄
- Payer Agent (insurance verification) 📋

---

## 🎯 Roadmap

### Immediate
- [ ] PayerAgent implementation
- [ ] More few-shot examples for edge cases
- [ ] Telephony integration (Twilio)

### Short-term
- [ ] Multi-language support (Arabic, Hindi)
- [ ] Kafka integration for enterprise scale
- [ ] HTML test client

### Long-term
- [ ] On-premise deployment guide
- [ ] HIPAA compliance certification
- [ ] EHR integrations (Epic, Cerner)

---

## 📚 References

### Industry Patterns
- [Retell AI Architecture](https://www.retellai.com/)
- [Vapi Voice Agents](https://vapi.ai/)
- [Amazon Alexa Intent Classification](https://developer.amazon.com/en-US/docs/alexa/custom-skills/understanding-how-users-invoke-custom-skills.html)

### Technical
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Ollama](https://ollama.ai/)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Adnan** - Data Engineering Master's Student (Dec 2025)

Building production-grade AI systems for healthcare. Open to Data Engineering and AI Engineering roles at FAANG/Big Tech.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/mohammedadnansohail1-pixel)
