"""
LLM Response Generator for InboundAgent.

Integrates:
- State-specific prompts
- Existing guardrails
- Fallback templates
"""

from typing import Optional, Any
from dataclasses import dataclass

from ...logging import get_logger
from ..guardrails import (
    ConversationGuardrails,
    GuardrailAction,
    GuardrailResult,
)
from .prompts import STATE_PROMPTS, StatePrompt, BASE_SYSTEM

logger = get_logger("agent.inbound.response")


@dataclass
class GeneratedResponse:
    """Result from response generation."""
    text: str
    source: str  # "llm" | "fallback" | "guardrail"
    guardrail_action: Optional[GuardrailAction] = None
    guardrail_reason: Optional[str] = None


class ResponseGenerator:
    """
    Generates natural LLM responses with guardrails and fallbacks.
    
    Flow:
    1. Check input guardrails (emergency, transfer, crisis)
    2. Generate LLM response with state-specific prompt
    3. Check output guardrails
    4. Fall back to template if LLM fails
    """
    
    def __init__(
        self,
        llm: Any,
        clinic_name: str = "Sunrise Medical",
        max_turns: int = 20,
    ):
        self.llm = llm
        self.clinic_name = clinic_name
        self.guardrails = ConversationGuardrails(
            clinic_name=clinic_name,
            max_turns=max_turns,
        )
        self._turn_count = 0
    
    def check_input(self, user_input: str, slots: dict) -> GuardrailResult:
        """Check user input against guardrails before processing."""
        return self.guardrails.check(user_input, slots)
    
    def generate(
        self,
        prompt_key: str,
        context: dict,
        user_input: Optional[str] = None,
    ) -> GeneratedResponse:
        """
        Generate a response for the given state.
        
        Args:
            prompt_key: Key from STATE_PROMPTS (e.g., "consent_confirmed")
            context: Template variables (patient_first_name, clinic_name, etc.)
            user_input: Optional user input for context
        
        Returns:
            GeneratedResponse with text and source
        """
        self._turn_count += 1
        
        # Get prompt config
        prompt_config = STATE_PROMPTS.get(prompt_key)
        if not prompt_config:
            logger.warning("unknown_prompt_key", key=prompt_key)
            return GeneratedResponse(
                text="How can I help you?",
                source="fallback",
            )
        
        # Add clinic_name to context
        context = {**context, "clinic_name": self.clinic_name}
        
        # Try LLM generation
        if self.llm:
            try:
                response_text = self._generate_llm(prompt_config, context, user_input)
                if response_text:
                    # Validate output
                    output_check = self.guardrails.check_output(response_text, context)
                    if output_check.action == GuardrailAction.CONTINUE:
                        return GeneratedResponse(text=response_text, source="llm")
                    else:
                        logger.warning(
                            "output_guardrail_triggered",
                            reason=output_check.reason,
                        )
            except Exception as e:
                logger.error("llm_generation_failed", error=str(e))
        
        # Fallback to template
        fallback_text = self._format_fallback(prompt_config.fallback, context)
        return GeneratedResponse(text=fallback_text, source="fallback")
    
    def generate_with_guardrails(
        self,
        prompt_key: str,
        context: dict,
        user_input: str,
        current_slots: dict,
    ) -> GeneratedResponse:
        """
        Full pipeline: check guardrails → generate → validate.
        
        Returns guardrail response if triggered, otherwise generated response.
        """
        # Check input guardrails
        guardrail_result = self.check_input(user_input, current_slots)
        
        if guardrail_result.action != GuardrailAction.CONTINUE:
            logger.info(
                "guardrail_triggered",
                action=guardrail_result.action.value,
                reason=guardrail_result.reason,
            )
            
            # Use guardrail response if provided
            if guardrail_result.response:
                return GeneratedResponse(
                    text=guardrail_result.response,
                    source="guardrail",
                    guardrail_action=guardrail_result.action,
                    guardrail_reason=guardrail_result.reason,
                )
        
        # Normal generation
        return self.generate(prompt_key, context, user_input)
    
    def _generate_llm(
        self,
        prompt_config: StatePrompt,
        context: dict,
        user_input: Optional[str],
    ) -> Optional[str]:
        """Generate response using LLM."""
        # Build context string
        context_str = self._build_context_string(context)
        
        # Format system prompt
        system_prompt = prompt_config.system_prompt.format(
            clinic_name=self.clinic_name,
            context=context_str,
        )
        
        # Format user prompt
        user_prompt = prompt_config.user_prompt_template.format(
            user_input=user_input or "",
            **context,
        )
        
        # Call LLM
        from ...core.types import LLMMessage
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = self.llm.generate(messages)
        
        # Clean response
        text = response.content.strip()
        
        # Remove any JSON if LLM returns it
        if text.startswith("{"):
            import json
            try:
                data = json.loads(text)
                text = data.get("response", text)
            except json.JSONDecodeError:
                pass
        
        # Remove quotes if wrapped
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        return text if text else None
    
    def _build_context_string(self, context: dict) -> str:
        """Build readable context for system prompt."""
        parts = []
        
        if context.get("patient_first_name"):
            parts.append(f"Patient: {context['patient_first_name']}")
        
        if context.get("visit_reason"):
            parts.append(f"Reason: {context['visit_reason']}")
        
        if context.get("preferred_day"):
            parts.append(f"Day: {context['preferred_day']}")
        
        if context.get("preferred_time"):
            parts.append(f"Time: {context['preferred_time']}")
        
        if context.get("available_days"):
            parts.append(f"Available days: {context['available_days']}")
        
        if context.get("available_times"):
            parts.append(f"Available times: {context['available_times']}")
        
        return "\n".join(parts) if parts else "No context yet"
    
    def _format_fallback(self, template: str, context: dict) -> str:
        """Format fallback template, handling missing keys."""
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning("fallback_missing_key", key=str(e))
            # Try partial format
            import re
            result = template
            for key, value in context.items():
                result = result.replace(f"{{{key}}}", str(value) if value else "")
            # Remove remaining placeholders
            result = re.sub(r"\{[^}]+\}", "", result)
            return result.strip()
    
    def reset(self):
        """Reset turn counter and guardrails."""
        self._turn_count = 0
        self.guardrails.turn_count = 0
        self.guardrails.unproductive_turns = 0
