"""Load flows from YAML files."""
from pathlib import Path
from typing import Union

import yaml

from .models import Flow, FlowState, StateType, Intent, Slot


def load_flow(path: Union[str, Path]) -> Flow:
    """
    Load a flow from a YAML file.
    
    Args:
        path: Path to the YAML flow file
        
    Returns:
        Parsed Flow object
    """
    path = Path(path)
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    return parse_flow(data)


def parse_flow(data: dict) -> Flow:
    """Parse a flow from a dictionary."""
    # Parse states
    states = {}
    for state_name, state_data in data.get("states", {}).items():
        states[state_name] = _parse_state(state_name, state_data)
    
    return Flow(
        name=data.get("name", "unnamed"),
        version=data.get("version", "1.0"),
        description=data.get("description"),
        initial_state=data.get("initial_state", "start"),
        states=states,
        global_fallback=data.get("global_fallback", "I'm sorry, something went wrong."),
        context_defaults=data.get("context_defaults", {}),
    )


def _parse_state(name: str, data: dict) -> FlowState:
    """Parse a single state."""
    # Parse intents
    intents = {}
    for intent_name, intent_data in data.get("intents", {}).items():
        intents[intent_name] = Intent(
            patterns=intent_data.get("patterns", []),
            examples=intent_data.get("examples", []),
            next=intent_data.get("next", ""),
        )
    
    # Parse slots
    slots = []
    for slot_data in data.get("slots", []):
        slots.append(Slot(
            name=slot_data.get("name", ""),
            type=slot_data.get("type", "string"),
            prompt=slot_data.get("prompt"),
            required=slot_data.get("required", True),
            validation=slot_data.get("validation"),
        ))
    
    return FlowState(
        name=name,
        type=StateType(data.get("type", "speak")),
        message=data.get("message"),
        intents=intents,
        slots=slots,
        fallback_message=data.get("fallback_message", "I didn't catch that. Could you repeat?"),
        max_retries=data.get("max_retries", 2),
        action=data.get("action"),
        action_params=data.get("action_params", {}),
        on_success=data.get("on_success"),
        on_failure=data.get("on_failure"),
        condition=data.get("condition"),
        if_true=data.get("if_true"),
        if_false=data.get("if_false"),
        next=data.get("next"),
    )
