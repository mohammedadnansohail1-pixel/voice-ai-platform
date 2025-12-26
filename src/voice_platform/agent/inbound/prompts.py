"""
State-specific prompts for InboundAgent LLM response generation.

VOICE AI GUIDELINES:
- Keep responses SHORT (1-2 sentences, max 30 words)
- Natural conversational tone
- No fluff or filler words
- Direct and clear
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StatePrompt:
    """Prompt configuration for a state."""
    system_prompt: str
    user_prompt_template: str
    fallback: str
    max_tokens: int = 60  # Keep short for voice


# =============================================================================
# System Prompt Base
# =============================================================================

BASE_SYSTEM = """You are a friendly medical receptionist at {clinic_name}.

CRITICAL RULES FOR VOICE AI:
- Maximum 1-2 sentences (under 30 words)
- Be warm but VERY concise
- No filler phrases ("I'd be happy to", "Let me help you with")
- Get straight to the point
- Sound natural when spoken aloud

Context: {context}"""


# =============================================================================
# State-Specific Prompts  
# =============================================================================

STATE_PROMPTS = {
    "greeting": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Generate a brief greeting that:
1. Thanks them for calling {clinic_name}
2. Mentions call recording
3. Asks for consent

MAX 2 sentences.""",
        fallback="Thank you for calling {clinic_name}. This call may be recorded for quality purposes. Do I have your consent to collect some information?",
    ),
    
    "collecting_consent": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""User response was unclear: "{user_input}"
Ask again for consent. ONE sentence only.""",
        fallback="I need your consent to continue. Do you agree? Yes or no?",
    ),
    
    "consent_confirmed": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Consent given. Ask for their full name. ONE sentence.""",
        fallback="Thank you. May I have your full name please?",
    ),
    
    "collecting_name": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Couldn't understand name from: "{user_input}"
Ask them to repeat. ONE sentence.""",
        fallback="I didn't catch that. Could you say your full name again?",
    ),
    
    "name_collected": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Got name: {patient_first_name}. Ask for date of birth. ONE sentence, use their first name.""",
        fallback="Thanks {patient_first_name}. What's your date of birth?",
    ),
    
    "collecting_dob": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Couldn't parse DOB from: "{user_input}"
Ask again with format hint. ONE sentence.""",
        fallback="Please say your date of birth, like March 15, 1985.",
    ),
    
    "dob_collected": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Got DOB. Ask for phone number. ONE sentence.""",
        fallback="Got it. What's the best phone number to reach you?",
    ),
    
    "collecting_phone": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Couldn't parse phone from: "{user_input}"
Ask for 10-digit number. ONE sentence.""",
        fallback="Please say your 10-digit phone number.",
    ),
    
    "phone_collected": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Got phone. Ask what brings them in. ONE sentence.""",
        fallback="Thanks! What brings you in today?",
    ),
    
    "collecting_reason": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Ask what brings them in. ONE sentence.""",
        fallback="What brings you in today?",
    ),
    
    "reason_collected": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""They said: {visit_reason}. Show empathy briefly, then offer days: {available_days}. TWO sentences max.""",
        fallback="Sorry to hear about your {visit_reason}. We have {available_days} available. Which works?",
    ),
    
    "collecting_day": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Ask which day. Available: {available_days}. ONE sentence.""",
        fallback="Which day works for you? We have {available_days}.",
    ),
    
    "day_unavailable": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""That day isn't available. Offer: {available_days}. ONE sentence.""",
        fallback="Sorry, that day's full. We have {available_days}. Which works?",
    ),
    
    "confirming_day": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Confirm day: {preferred_day}. ONE sentence, yes/no question.""",
        fallback="Just to confirm, {preferred_day}?",
    ),
    
    "day_confirmed": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Day confirmed: {preferred_day}. Offer times: {available_times}. ONE sentence.""",
        fallback="Great! We have {available_times} on {preferred_day}. Which time?",
    ),
    
    "collecting_time": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Ask which time. Available: {available_times}. ONE sentence.""",
        fallback="Which time works? We have {available_times}.",
    ),
    
    "time_unavailable": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""That time's taken. Offer: {available_times}. ONE sentence.""",
        fallback="That time's not available. We have {available_times}.",
    ),
    
    "confirming": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Confirm booking: {preferred_day} at {preferred_time} for {visit_reason}. Ask yes/no. TWO sentences max.""",
        fallback="{preferred_day} at {preferred_time} for your {visit_reason}. Should I book it?",
    ),
    
    "booking_complete": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Booking confirmed for {patient_first_name}:
- Day: {preferred_day}
- Time: {preferred_time}  
- Confirmation: {confirmation_number}
Give warm closing with confirmation number. TWO sentences.""",
        fallback="You're all set, {patient_first_name}! {preferred_day} at {preferred_time}, confirmation {confirmation_number}. See you then!",
    ),
    
    "transfer": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""Transfer to human. Brief, reassuring. ONE sentence.""",
        fallback="Let me connect you with someone who can help. Please hold.",
    ),
    
    "consent_refused": StatePrompt(
        system_prompt=BASE_SYSTEM,
        user_prompt_template="""They declined consent. Be understanding, offer help. ONE sentence.""",
        fallback="I understand. Is there anything else I can help with?",
    ),
}


def get_prompt(state_key: str) -> Optional[StatePrompt]:
    """Get prompt for a state."""
    return STATE_PROMPTS.get(state_key)


def format_fallback(state_key: str, **kwargs) -> str:
    """Format fallback template with context."""
    prompt = STATE_PROMPTS.get(state_key)
    if prompt:
        try:
            return prompt.fallback.format(**kwargs)
        except KeyError:
            return prompt.fallback
    return "How can I help you?"
