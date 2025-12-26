"""Conversation guardrails for safety and quality."""
import re
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class GuardrailAction(Enum):
    CONTINUE = "continue"           # Normal flow
    TRANSFER = "transfer"           # Transfer to human
    EMERGENCY = "emergency"         # Medical emergency
    CLARIFY = "clarify"            # Ask user to repeat
    REDIRECT = "redirect"          # Redirect to scheduling

@dataclass
class GuardrailResult:
    action: GuardrailAction
    reason: str
    response: Optional[str] = None  # Override response if needed

class ConversationGuardrails:
    """Safety and quality guardrails for conversations."""
    
    # Medical emergencies - MUST transfer/911
    EMERGENCY_PATTERNS = [
        r"chest pain",
        r"can'?t breathe",
        r"difficulty breathing",
        r"heart attack",
        r"stroke",
        r"unconscious",
        r"seizure",
        r"severe bleeding",
        r"choking",
        r"allergic reaction",
        r"anaphyla",
        r"overdose",
    ]
    
    # Dental emergencies - prioritize but don't 911
    DENTAL_EMERGENCY_PATTERNS = [
        r"knocked out.{0,10}tooth",
        r"tooth.{0,10}knocked out",
        r"avulsed",
        r"severe.{0,10}swelling",
        r"abscess",
        r"can'?t open.{0,10}mouth",
        r"jaw.{0,10}(locked|broken|fractured)",
        r"blood.{0,10}everywhere",
        r"won'?t stop bleeding",
    ]
    
    # Transfer triggers
    TRANSFER_PATTERNS = [
        r"speak.{0,10}(human|person|someone|agent|real)",
        r"talk.{0,10}(human|person|someone|agent|real)",
        r"transfer.{0,10}me",
        r"real person",
        r"actual person",
        r"representative",
        r"operator",
        r"receptionist",
    ]
    
    # Frustration indicators
    FRUSTRATION_PATTERNS = [
        r"this is (ridiculous|stupid|useless|frustrating)",
        r"(hate|can'?t stand).{0,10}(this|you|ai|bot)",
        r"(stupid|dumb|useless).{0,10}(bot|ai|machine)",
        r"(forget|screw|f\*ck) (it|this)",
        r"i give up",
        r"this (doesn'?t|isn'?t) (work|help)",
        r"waste.{0,10}time",
    ]
    
    # Crisis/self-harm - handle with care
    CRISIS_PATTERNS = [
        r"(want to|going to|thinking about).{0,10}(kill|hurt|harm).{0,10}(myself|self)",
        r"suicide",
        r"end.{0,10}(my life|it all)",
        r"don'?t want to (live|be here|exist)",
    ]
    
    # Off-topic but harmless
    OFF_TOPIC_PATTERNS = [
        r"(what'?s|how'?s).{0,10}weather",
        r"(order|want).{0,10}(pizza|food|coffee)",
        r"(who|what).{0,10}(president|king|prime minister)",
        r"tell.{0,10}(joke|story)",
        r"(play|sing).{0,10}(music|song)",
    ]
    
    def __init__(self, clinic_name: str = "our clinic", max_turns: int = 15):
        self.clinic_name = clinic_name
        self.max_turns = max_turns
        self.turn_count = 0
        self.unproductive_turns = 0
    
    def check(self, user_input: str, current_slots: dict) -> GuardrailResult:
        """Check user input against all guardrails."""
        text = user_input.lower().strip()
        
        # 1. Medical Emergency - highest priority
        if self._matches_any(text, self.EMERGENCY_PATTERNS):
            return GuardrailResult(
                action=GuardrailAction.EMERGENCY,
                reason="medical_emergency",
                response="I'm concerned about what you're describing. This sounds like a medical emergency. Please call 911 or go to your nearest emergency room immediately. Your health and safety come first."
            )
        
        # 2. Dental Emergency - urgent but not 911
        if self._matches_any(text, self.DENTAL_EMERGENCY_PATTERNS):
            return GuardrailResult(
                action=GuardrailAction.TRANSFER,
                reason="dental_emergency", 
                response="This sounds like a dental emergency that needs immediate attention. Let me transfer you to our staff right away so we can get you in as soon as possible. Please hold."
            )
        
        # 3. Crisis/Self-harm - compassionate response
        if self._matches_any(text, self.CRISIS_PATTERNS):
            return GuardrailResult(
                action=GuardrailAction.TRANSFER,
                reason="crisis_detected",
                response="I hear that you're going through a really difficult time. I want to make sure you get the right support. The National Suicide Prevention Lifeline is available 24/7 at 988. Would you like me to transfer you to someone who can help?"
            )
        
        # 4. Explicit transfer request
        if self._matches_any(text, self.TRANSFER_PATTERNS):
            return GuardrailResult(
                action=GuardrailAction.TRANSFER,
                reason="transfer_requested",
                response="Of course, I'll transfer you to a staff member right away. Please hold for just a moment."
            )
        
        # 5. Frustration detection
        if self._matches_any(text, self.FRUSTRATION_PATTERNS):
            return GuardrailResult(
                action=GuardrailAction.TRANSFER,
                reason="user_frustrated",
                response="I apologize for any frustration. Let me transfer you to one of our team members who can assist you directly. Please hold."
            )
        
        # 6. Check for unproductive conversation
        self.turn_count += 1
        if self._is_unproductive_input(text):
            self.unproductive_turns += 1
        else:
            self.unproductive_turns = 0
        
        if self.unproductive_turns >= 4:
            return GuardrailResult(
                action=GuardrailAction.TRANSFER,
                reason="unproductive_conversation",
                response="I want to make sure we get you the help you need. Let me transfer you to a staff member who can assist you better. Please hold."
            )
        
        # 7. Max turns reached
        if self.turn_count >= self.max_turns:
            return GuardrailResult(
                action=GuardrailAction.TRANSFER,
                reason="max_turns_reached",
                response="I want to make sure we complete your request. Let me transfer you to a staff member to finish up. Please hold."
            )
        
        # 8. Off-topic (harmless) - redirect
        if self._matches_any(text, self.OFF_TOPIC_PATTERNS):
            return GuardrailResult(
                action=GuardrailAction.REDIRECT,
                reason="off_topic",
                response=None  # Let LLM handle redirect naturally
            )
        
        return GuardrailResult(action=GuardrailAction.CONTINUE, reason="ok")
    
    def check_output(self, response: str, slots: dict) -> GuardrailResult:
        """Validate LLM output before sending to user."""
        
        # Check for hallucinated slots
        if slots.get("appointment_day") and slots.get("appointment_time"):
            day = slots["appointment_day"].lower()
            time = slots["appointment_time"].lower()
            
            # Detect nonsensical combinations
            if day in time or time in day:
                return GuardrailResult(
                    action=GuardrailAction.CLARIFY,
                    reason="invalid_slot_combination",
                    response="I apologize for the confusion. Let me start over - what day and time work best for you?"
                )
        
        # Check for "no reason mentioned" type values
        if slots.get("visit_reason"):
            reason = slots["visit_reason"].lower()
            invalid_reasons = ["no reason", "unknown", "not mentioned", "null", "..."]
            if any(inv in reason for inv in invalid_reasons):
                return GuardrailResult(
                    action=GuardrailAction.CONTINUE,
                    reason="invalid_reason_cleared",
                    response=None
                )
        
        return GuardrailResult(action=GuardrailAction.CONTINUE, reason="ok")
    
    def _matches_any(self, text: str, patterns: list) -> bool:
        """Check if text matches any pattern."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _is_unproductive_input(self, text: str) -> bool:
        """Check if input is unproductive (too short, vague, etc.)"""
        unproductive = [
            r"^(um+|uh+|hmm+|ah+)$",
            r"^(ok|okay|sure|yeah|yes|no|maybe|whatever|idk|i don'?t know)$",
            r"^(hi|hello|hey)$",
            r"^.{0,3}$",  # Very short
        ]
        return self._matches_any(text, unproductive)


def detect_slot_reset(user_input: str) -> bool:
    """Detect if user wants to change/reset all slots."""
    reset_patterns = [
        r"(actually|wait|no).{0,20}(change|different|start over|completely)",
        r"(let'?s|can we).{0,10}start (over|fresh|again)",
        r"(forget|ignore).{0,10}(that|what i said|everything)",
        r"(change|switch).{0,10}(everything|all of)",
    ]
    text = user_input.lower()
    for pattern in reset_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def detect_cancellation_intent(user_input: str) -> bool:
    """Detect if user wants to cancel (not schedule) an appointment."""
    cancel_patterns = [
        r"(cancel|canceling|cancellation)",
        r"(need to|want to|have to).{0,10}cancel",
        r"can'?t (make|come to|attend).{0,10}appointment",
    ]
    text = user_input.lower()
    for pattern in cancel_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
