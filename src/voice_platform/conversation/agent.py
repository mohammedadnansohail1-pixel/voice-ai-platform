"""LLM-centric conversation agent with production guardrails."""
import json
import re
from typing import Optional

from .schemas import ConversationState, AgentResponse, Stage
from .guardrails import (
    ConversationGuardrails, 
    GuardrailAction, 
    detect_slot_reset,
    detect_cancellation_intent
)
from ..logging import get_logger

logger = get_logger("conversation.agent")

SYSTEM_PROMPT = """You are a friendly AI receptionist for {clinic_name}. Your goal is to efficiently book ONE appointment while being warm and helpful.

CLINIC INFO:
- Name: {clinic_name}
- Address: 123 Main Street, Suite 100
- Hours: Monday-Friday 8am-5pm, Saturday 9am-1pm
- Phone: (555) 123-4567

INFORMATION TO COLLECT:
1. Visit reason (what brings them in today)
2. ONE specific day preference  
3. ONE specific time preference

CRITICAL RULES:
- "I work mornings" = ONLY offer AFTERNOON slots
- "I work afternoons" = ONLY offer MORNING slots
- When offering slots, give exactly 2 specific options
- If user says "yes" without picking, ask which option
- NEVER make up times - only offer from: {available_slots}

AVAILABLE SLOTS:
{available_slots}

CURRENT STATE:
{state_context}

Respond with JSON only:
{{"extracted_slots": {{"visit_reason": "...", "appointment_day": "...", "appointment_time": "..."}}, "response": "your reply", "ready_to_confirm": false, "availability_preference": "morning/afternoon/any"}}"""

GREETING_MESSAGE = "Hi, thank you for calling {clinic_name}. This is an AI assistant and I can help you schedule an appointment. How can I help you today?"


class ConversationAgent:
    """LLM-powered conversation agent with guardrails."""
    
    def __init__(self, llm, clinic_name: str = "Sunrise Dental"):
        self.llm = llm
        self.clinic_name = clinic_name
        self.state = ConversationState()
        self.required_slots = ["visit_reason", "appointment_day", "appointment_time"]
        self.guardrails = ConversationGuardrails(clinic_name=clinic_name)
        
        # Available slots - in production from PMS API
        self.available_slots = [
            ("Tuesday", "2:00 PM"),
            ("Tuesday", "4:30 PM"),
            ("Wednesday", "10:00 AM"),
            ("Wednesday", "3:00 PM"),
            ("Thursday", "9:00 AM"),
            ("Thursday", "3:00 PM"),
            ("Friday", "11:00 AM"),
            ("Friday", "2:30 PM"),
        ]
    
    def _get_slots_for_preference(self, preference: str, day_filter: str = None) -> list:
        """Filter slots by time preference and day."""
        filtered = []
        for day, time in self.available_slots:
            if day_filter and day_filter.lower() not in day.lower():
                continue
            
            hour = int(time.split(":")[0])
            is_pm = "PM" in time
            if is_pm and hour != 12:
                hour += 12
            elif not is_pm and hour == 12:
                hour = 0
            
            is_afternoon = hour >= 12
            
            if preference == "afternoon" and not is_afternoon:
                continue
            if preference == "morning" and is_afternoon:
                continue
            
            filtered.append((day, time))
        
        return filtered if filtered else self.available_slots[:4]
    
    def start(self) -> AgentResponse:
        self.state.stage = Stage.GREETING
        greeting = GREETING_MESSAGE.format(clinic_name=self.clinic_name)
        self.state.history.append({"role": "assistant", "content": greeting})
        logger.info("conversation_started", clinic=self.clinic_name)
        return AgentResponse(
            message=greeting,
            slots={},
            stage=Stage.GREETING,
        )
    
    def process(self, user_input: str) -> AgentResponse:
        logger.info("user_input", text=user_input, stage=self.state.stage.value)
        
        # === GUARDRAILS CHECK (BEFORE LLM) ===
        guardrail_result = self.guardrails.check(user_input, self.state.slots)
        
        if guardrail_result.action == GuardrailAction.EMERGENCY:
            logger.warning("guardrail_triggered", reason=guardrail_result.reason)
            self.state.stage = Stage.DONE
            return AgentResponse(
                message=guardrail_result.response,
                slots=dict(self.state.slots),
                stage=Stage.DONE,
                ended=True,
            )
        
        if guardrail_result.action == GuardrailAction.TRANSFER:
            logger.info("guardrail_transfer", reason=guardrail_result.reason)
            self.state.stage = Stage.DONE
            return AgentResponse(
                message=guardrail_result.response,
                slots=dict(self.state.slots),
                stage=Stage.DONE,
                ended=True,
            )
        
        # === SPECIAL INTENTS ===
        
        # Check for cancellation (different flow)
        if detect_cancellation_intent(user_input) and not self.state.slots:
            logger.info("cancellation_detected")
            return AgentResponse(
                message="I can help you with cancellations. Let me transfer you to our scheduling team who can look up your appointment. Please hold.",
                slots={},
                stage=Stage.DONE,
                ended=True,
            )
        
        # Check for slot reset request
        if detect_slot_reset(user_input):
            logger.info("slot_reset_detected")
            # Keep only the new info, clear old slots
            self.state.slots = {}
            self.state.stage = Stage.COLLECTING
        
        # === NORMAL PROCESSING ===
        self.state.history.append({"role": "user", "content": user_input})
        
        if self.state.stage == Stage.GREETING:
            self.state.stage = Stage.COLLECTING
        
        # Detect availability preference
        input_lower = user_input.lower()
        if any(p in input_lower for p in ["work morning", "busy morning", "only afternoon", "afternoons work"]):
            self.state.slots["availability_preference"] = "afternoon"
        elif any(p in input_lower for p in ["work afternoon", "busy afternoon", "only morning", "mornings work"]):
            self.state.slots["availability_preference"] = "morning"
        
        # Get filtered slots
        preference = self.state.slots.get("availability_preference", "any")
        day_filter = self.state.slots.get("appointment_day")
        matching_slots = self._get_slots_for_preference(preference, day_filter)
        slots_str = ", ".join([f"{d} at {t}" for d, t in matching_slots[:4]])
        
        # Build prompt
        system = SYSTEM_PROMPT.format(
            clinic_name=self.clinic_name,
            state_context=self.state.to_context(),
            available_slots=slots_str
        )
        
        recent_history = self.state.history[-6:]
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in recent_history])
        
        prompt = f"""Conversation:
{history_text}

User just said: "{user_input}"

Respond with JSON:"""
        
        # Call LLM
        try:
            raw_response = self.llm.generate(prompt, system)
            logger.debug("llm_raw", response=raw_response[:200])
            result = self._parse_response(raw_response)
        except Exception as e:
            logger.error("llm_error", error=str(e))
            result = {
                "extracted_slots": {},
                "response": "I'm sorry, I didn't catch that. Could you repeat?",
                "ready_to_confirm": False,
            }
        
        # Update slots
        new_slots = result.get("extracted_slots", {})
        if isinstance(new_slots, dict):
            for key, value in new_slots.items():
                if value and value not in ["null", "...", "", "unknown", "no reason mentioned", "no reason"]:
                    value_str = str(value).strip()
                    if not value_str:
                        continue
                    
                    if isinstance(value, list):
                        value_str = ", ".join(str(v) for v in value)
                    
                    if key == "appointment_day":
                        value_str = self._normalize_day(value_str)
                        if not value_str:
                            continue
                    
                    if key == "appointment_time":
                        pref = self.state.slots.get("availability_preference", "any")
                        if not self._time_matches_preference(value_str, pref):
                            logger.warning("time_conflicts", time=value_str, pref=pref)
                            continue
                    
                    self.state.slots[key] = value_str
                    logger.info("slot_extracted", slot=key, value=value_str)
        
        # === OUTPUT GUARDRAILS ===
        output_check = self.guardrails.check_output(
            result.get("response", ""),
            self.state.slots
        )
        
        if output_check.action == GuardrailAction.CLARIFY:
            logger.warning("output_guardrail", reason=output_check.reason)
            self.state.slots.pop("appointment_day", None)
            self.state.slots.pop("appointment_time", None)
            return AgentResponse(
                message=output_check.response,
                slots=dict(self.state.slots),
                stage=Stage.COLLECTING,
            )
        
        message = result.get("response", "What day works best for you?")
        
        # Check completion
        missing = self.state.get_missing_slots(self.required_slots)
        
        if not missing and self.state.stage == Stage.COLLECTING:
            self.state.stage = Stage.CONFIRMING
        
        # Handle confirmation stage
        if self.state.stage == Stage.CONFIRMING:
            day = self.state.slots.get('appointment_day', '')
            time = self.state.slots.get('appointment_time', '')
            reason = self.state.slots.get('visit_reason', '')
            
            if not missing:
                message = f"Perfect! Let me confirm: {day} at {time} for {reason}. Is that correct?"
            
            if self._user_confirmed(user_input):
                self.state.stage = Stage.DONE
                self.state.confirmed = True
                message = f"You're all set for {day} at {time}. You'll receive a text confirmation shortly. Thank you for calling {self.clinic_name}!"
        
        self.state.history.append({"role": "assistant", "content": message})
        
        ended = self.state.stage == Stage.DONE
        
        logger.info("response_generated", 
                   stage=self.state.stage.value, 
                   slots={k:v for k,v in self.state.slots.items() if k != 'availability_preference'},
                   ended=ended)
        
        return AgentResponse(
            message=message,
            slots=dict(self.state.slots),
            stage=self.state.stage,
            ready_to_book=not missing,
            ended=ended,
        )
    
    def _time_matches_preference(self, time_str: str, preference: str) -> bool:
        if preference == "any":
            return True
        
        try:
            hour_match = re.search(r'(\d{1,2})', time_str)
            if not hour_match:
                return True
            hour = int(hour_match.group(1))
            
            is_pm = "pm" in time_str.lower()
            if is_pm and hour != 12:
                hour += 12
            elif not is_pm and hour == 12:
                hour = 0
            
            is_afternoon = hour >= 12
            
            if preference == "afternoon":
                return is_afternoon
            elif preference == "morning":
                return not is_afternoon
        except:
            pass
        
        return True
    
    def _normalize_day(self, value: str) -> Optional[str]:
        value_lower = value.lower()
        days = {
            "monday": "Monday", "mon": "Monday",
            "tuesday": "Tuesday", "tue": "Tuesday", "tues": "Tuesday",
            "wednesday": "Wednesday", "wed": "Wednesday",
            "thursday": "Thursday", "thu": "Thursday", "thurs": "Thursday",
            "friday": "Friday", "fri": "Friday",
            "saturday": "Saturday", "sat": "Saturday",
            "sunday": "Sunday", "sun": "Sunday",
            "today": "Today",
            "tomorrow": "Tomorrow",
        }
        
        for key, normalized in days.items():
            if key in value_lower:
                return normalized
        return None
    
    def _parse_response(self, raw: str) -> dict:
        raw = raw.strip()
        
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
        
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        try:
            start = raw.find('{')
            if start != -1:
                depth = 0
                end = start
                for i, c in enumerate(raw[start:], start):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                
                return json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            logger.warning("json_parse_failed", error=str(e))
        
        return {
            "extracted_slots": {},
            "response": raw.strip()[:200] if raw else "Could you tell me more?",
            "ready_to_confirm": False,
        }
    
    def _user_confirmed(self, text: str) -> bool:
        confirms = ["yes", "yeah", "correct", "right", "confirm", "book", "sounds good", 
                   "perfect", "ok", "okay", "sure", "yep", "that works", "great", "that's right"]
        denies = ["no", "nope", "wrong", "change", "actually", "wait", "different", "not"]
        text_lower = text.lower()
        
        if any(d in text_lower for d in denies):
            return False
        
        return any(c in text_lower for c in confirms)
