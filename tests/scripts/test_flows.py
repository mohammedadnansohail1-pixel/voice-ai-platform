"""Test flow engine."""
import sys
sys.path.insert(0, 'src')

from rich.console import Console

console = Console()

def test_section(name: str):
    console.print(f"\n[bold cyan]{'='*50}[/]")
    console.print(f"[bold cyan]Testing: {name}[/]")
    console.print(f"[bold cyan]{'='*50}[/]")

def test_pass(msg: str):
    console.print(f"  [green]✓[/] {msg}")

def test_fail(msg: str, error: Exception):
    console.print(f"  [red]✗[/] {msg}")
    console.print(f"    [red]{type(error).__name__}: {error}[/]")

# =============================================================================
# Test 1: Imports
# =============================================================================
test_section("Flow Imports")

try:
    from voice_platform.flows import Flow, FlowState, FlowEngine, load_flow, Intent, Slot
    from voice_platform.flows.models import StateType
    from voice_platform.flows.engine import FlowContext, EngineResponse
    from voice_platform.logging import setup_logging
    from voice_platform.core.config import LoggingConfig
    
    setup_logging(LoggingConfig(level="WARNING", format="console"))
    test_pass("All imports successful")
except Exception as e:
    test_fail("Imports", e)
    sys.exit(1)

# =============================================================================
# Test 2: Load YAML Flow
# =============================================================================
test_section("Load YAML Flow")

try:
    flow = load_flow("configs/flows/appointment.yaml")
    
    assert flow.name == "appointment_scheduling"
    test_pass(f"Flow name: {flow.name}")
    
    assert flow.initial_state == "greeting"
    test_pass(f"Initial state: {flow.initial_state}")
    
    assert len(flow.states) > 0
    test_pass(f"States loaded: {len(flow.states)}")
    
    # Check state types
    assert flow.states["greeting"].type == StateType.SPEAK
    assert flow.states["detect_intent"].type == StateType.LISTEN
    assert flow.states["ask_reason"].type == StateType.SPEAK_LISTEN
    assert flow.states["check_availability"].type == StateType.ACTION
    assert flow.states["closing"].type == StateType.END
    test_pass("State types parsed correctly")
    
    # Check intents
    detect_state = flow.states["detect_intent"]
    assert "schedule" in detect_state.intents
    assert "appointment" in detect_state.intents["schedule"].patterns
    test_pass(f"Intents loaded: {list(detect_state.intents.keys())}")
    
    # Check context defaults
    assert flow.context_defaults["clinic_name"] == "Sunrise Medical Clinic"
    test_pass(f"Context defaults: {flow.context_defaults}")

except Exception as e:
    test_fail("Load YAML", e)
    raise

# =============================================================================
# Test 3: Flow Engine - Basic Execution
# =============================================================================
test_section("Flow Engine - Basic Execution")

try:
    flow = load_flow("configs/flows/appointment.yaml")
    engine = FlowEngine(flow)
    
    # Start the flow
    response = engine.start()
    assert response.message is not None
    assert "Sunrise Medical Clinic" in response.message  # Interpolation worked
    test_pass(f"Start message: '{response.message[:50]}...'")
    
    # Should be waiting for input (detect_intent is LISTEN)
    assert response.needs_input
    test_pass(f"Waiting for input at: {response.current_state}")

except Exception as e:
    test_fail("Basic execution", e)
    raise

# =============================================================================
# Test 4: Flow Engine - Intent Detection
# =============================================================================
test_section("Flow Engine - Intent Detection")

try:
    flow = load_flow("configs/flows/appointment.yaml")
    engine = FlowEngine(flow)
    
    response = engine.start()
    
    # User says they want to schedule
    response = engine.process_input("I'd like to schedule an appointment")
    
    assert response.current_state == "ask_reason"
    assert "reason" in response.message.lower()
    test_pass(f"Intent 'schedule' detected, moved to: {response.current_state}")
    
    # Test different intent
    engine2 = FlowEngine(flow)
    engine2.start()
    response = engine2.process_input("What are your hours?")
    
    assert response.current_state == "anything_else"  # tell_hours -> anything_else
    assert "Monday" in response.message or "8 AM" in response.message
    test_pass(f"Intent 'hours' detected, responded with hours")

except Exception as e:
    test_fail("Intent detection", e)
    raise

# =============================================================================
# Test 5: Flow Engine - Slot Filling
# =============================================================================
test_section("Flow Engine - Slot Filling")

try:
    flow = load_flow("configs/flows/appointment.yaml")
    engine = FlowEngine(flow)
    
    engine.start()
    engine.process_input("I need to book an appointment")
    
    # Fill visit_reason slot
    response = engine.process_input("I have a headache")
    assert engine.context.get_slot("visit_reason") == "I have a headache"
    test_pass(f"Slot 'visit_reason' filled: '{engine.context.get_slot('visit_reason')}'")
    
    # Fill preferred_day slot
    response = engine.process_input("Monday")
    assert engine.context.get_slot("preferred_day") == "Monday"
    test_pass(f"Slot 'preferred_day' filled: '{engine.context.get_slot('preferred_day')}'")
    
    # Fill preferred_time slot
    response = engine.process_input("morning")
    assert engine.context.get_slot("preferred_time") == "morning"
    test_pass(f"Slot 'preferred_time' filled: '{engine.context.get_slot('preferred_time')}'")
    
    # Should now be at action state
    assert response.action_request is not None
    assert response.action_request["action"] == "calendar.check_slots"
    test_pass(f"Action requested: {response.action_request['action']}")

except Exception as e:
    test_fail("Slot filling", e)
    raise

# =============================================================================
# Test 6: Flow Engine - Action Handling
# =============================================================================
test_section("Flow Engine - Action Handling")

try:
    flow = load_flow("configs/flows/appointment.yaml")
    engine = FlowEngine(flow)
    
    # Fast forward to action state
    engine.start()
    engine.process_input("schedule appointment")
    engine.process_input("checkup")
    engine.process_input("Tuesday")
    response = engine.process_input("afternoon")
    
    assert response.action_request is not None
    test_pass(f"At action state: {response.current_state}")
    
    # Simulate action success
    response = engine.execute_action_result(success=True, result={"slot": "2:00 PM"})
    assert response.current_state == "offer_slot"
    test_pass(f"Action success -> moved to: {response.current_state}")
    
    # Test action failure path
    engine2 = FlowEngine(flow)
    engine2.start()
    engine2.process_input("book")
    engine2.process_input("cold")
    engine2.process_input("Friday")
    engine2.process_input("morning")
    
    response = engine2.execute_action_result(success=False)
    assert response.current_state == "ask_preferred_day"  # no_availability -> ask_preferred_day
    test_pass(f"Action failure -> moved to: {response.current_state}")

except Exception as e:
    test_fail("Action handling", e)
    raise

# =============================================================================
# Test 7: Flow Engine - Complete Flow
# =============================================================================
test_section("Flow Engine - Complete Happy Path")

try:
    flow = load_flow("configs/flows/appointment.yaml")
    engine = FlowEngine(flow)
    
    console.print("  [dim]Simulating full conversation:[/]")
    
    # Start
    response = engine.start()
    console.print(f"  [blue]Bot:[/] {response.message}")
    
    # Schedule intent
    response = engine.process_input("I want to schedule an appointment")
    console.print(f"  [yellow]User:[/] I want to schedule an appointment")
    console.print(f"  [blue]Bot:[/] {response.message}")
    
    # Visit reason
    response = engine.process_input("Annual checkup")
    console.print(f"  [yellow]User:[/] Annual checkup")
    console.print(f"  [blue]Bot:[/] {response.message}")
    
    # Preferred day
    response = engine.process_input("Next Monday")
    console.print(f"  [yellow]User:[/] Next Monday")
    console.print(f"  [blue]Bot:[/] {response.message}")
    
    # Preferred time
    response = engine.process_input("Morning please")
    console.print(f"  [yellow]User:[/] Morning please")
    
    # Handle action
    response = engine.execute_action_result(success=True)
    console.print(f"  [dim](Action: calendar.check_slots)[/]")
    console.print(f"  [blue]Bot:[/] {response.message}")
    
    # Confirm
    response = engine.process_input("Yes, that works")
    console.print(f"  [yellow]User:[/] Yes, that works")
    console.print(f"  [blue]Bot:[/] {response.message}")
    
    # Check flow ended
    assert response.ended
    test_pass("Flow completed successfully")
    
    # Check slots collected
    console.print(f"  [dim]Slots: {response.slots}[/]")

except Exception as e:
    test_fail("Complete flow", e)
    raise

# =============================================================================
# Test 8: Fallback & Retries
# =============================================================================
test_section("Fallback & Retries")

try:
    flow = load_flow("configs/flows/appointment.yaml")
    engine = FlowEngine(flow)
    
    engine.start()
    
    # Say something that doesn't match any intent
    response = engine.process_input("blah blah blah")
    assert "schedule" in response.message.lower() or "cancel" in response.message.lower()
    test_pass(f"Fallback triggered: '{response.message[:50]}...'")
    
    # Retry counter
    assert engine.context.retry_count == 1
    test_pass(f"Retry count: {engine.context.retry_count}")
    
    # Keep failing
    engine.process_input("still nonsense")
    response = engine.process_input("more gibberish")
    
    # Should hit max retries and end
    assert response.ended
    test_pass(f"Max retries hit, flow ended with: '{response.message[:40]}...'")

except Exception as e:
    test_fail("Fallback & retries", e)
    raise

# =============================================================================
# Summary
# =============================================================================
console.print(f"\n[bold green]{'='*50}[/]")
console.print("[bold green]All flow engine tests passed![/]")
console.print(f"[bold green]{'='*50}[/]")
