#!/usr/bin/env python3
"""Test the tool-calling agent in CLI mode."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_platform.agent import ToolCallingAgent


def main():
    print("=" * 60)
    print("Tool-Calling Agent Test")
    print("=" * 60)
    print()
    
    # Create agent
    agent = ToolCallingAgent(clinic_name="Sunrise Medical")
    
    # Start conversation
    greeting = agent.start()
    print(f"Agent: {greeting}")
    print()
    
    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            
            # Process input
            response = agent.process(user_input)
            
            print(f"Agent: {response.text}")
            print(f"   [State: {agent.context.state.value} | Slots: {agent.context.slots_summary()}]")
            print()
            
            if response.ended:
                if response.booking:
                    print(f"Booking confirmed: {response.booking}")
                elif response.transfer:
                    print("Transferring to human...")
                break
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
