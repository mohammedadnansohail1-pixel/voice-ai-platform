"""Flow YAML loader."""
from pathlib import Path
from typing import Optional

import yaml

from .models import (
    Flow, 
    FlowStep, 
    FlowCondition, 
    SlotDefinition,
    ActionType,
    ConditionOperator,
)
from ..logging import get_logger

logger = get_logger("flows.loader")


def parse_slot(slot_data: dict | str) -> SlotDefinition:
    """Parse slot definition from YAML."""
    if isinstance(slot_data, str):
        return SlotDefinition(name=slot_data)
    
    return SlotDefinition(
        name=slot_data["name"],
        type=slot_data.get("type", "string"),
        prompt=slot_data.get("prompt"),
        required=slot_data.get("required", True),
        choices=slot_data.get("choices", []),
        patterns=slot_data.get("patterns", []),
        default=slot_data.get("default"),
    )


def parse_condition(cond_data: dict) -> FlowCondition:
    """Parse condition from YAML."""
    operator = ConditionOperator(cond_data.get("operator", "equals"))
    
    return FlowCondition(
        variable=cond_data["variable"],
        operator=operator,
        value=cond_data.get("value"),
        next_step=cond_data["next"],
    )


def parse_step(step_id: str, step_data: dict) -> FlowStep:
    """Parse step from YAML."""
    # Parse slots
    extract = []
    if "extract" in step_data:
        extract_data = step_data["extract"]
        if isinstance(extract_data, list):
            extract = [parse_slot(s) for s in extract_data]
        elif isinstance(extract_data, str):
            extract = [SlotDefinition(name=extract_data)]
        elif isinstance(extract_data, dict):
            extract = [parse_slot(extract_data)]
    
    # Parse conditions
    conditions = []
    if "conditions" in step_data:
        conditions = [parse_condition(c) for c in step_data["conditions"]]
    
    # Parse action
    action = None
    if "action" in step_data:
        action = ActionType(step_data["action"])
    
    return FlowStep(
        id=step_id,
        say=step_data.get("say"),
        listen=step_data.get("listen", False),
        extract=extract,
        conditions=conditions,
        next_step=step_data.get("next"),
        action=action,
        action_params=step_data.get("action_params", {}),
        retries=step_data.get("retries", 2),
        timeout_seconds=step_data.get("timeout", 10),
        on_timeout=step_data.get("on_timeout"),
        on_error=step_data.get("on_error"),
    )


def load_flow(path: str | Path) -> Flow:
    """Load flow from YAML file."""
    path = Path(path)
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    # Parse steps
    steps = {}
    for step_id, step_data in data.get("steps", {}).items():
        steps[step_id] = parse_step(step_id, step_data)
    
    # Parse global slots
    global_slots = []
    if "slots" in data:
        global_slots = [parse_slot(s) for s in data["slots"]]
    
    flow = Flow(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        version=data.get("version", "1.0"),
        initial_step=data.get("initial_step", "start"),
        steps=steps,
        global_slots=global_slots,
        fallback_step=data.get("fallback_step"),
        metadata=data.get("metadata", {}),
    )
    
    # Validate
    errors = flow.validate()
    if errors:
        logger.warning("flow_validation_errors", flow=flow.name, errors=errors)
    
    logger.info("flow_loaded", name=flow.name, steps=len(steps))
    return flow


def load_flows_from_directory(directory: str | Path) -> dict[str, Flow]:
    """Load all flows from a directory."""
    directory = Path(directory)
    flows = {}
    
    for path in directory.glob("*.yaml"):
        try:
            flow = load_flow(path)
            flows[flow.name] = flow
        except Exception as e:
            logger.error("flow_load_error", path=str(path), error=str(e))
    
    logger.info("flows_loaded", count=len(flows))
    return flows
