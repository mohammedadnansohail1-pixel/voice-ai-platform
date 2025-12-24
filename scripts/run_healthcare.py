#!/usr/bin/env python3
"""Run healthcare voice assistant."""
import sys
import argparse

sys.path.insert(0, "src")

from voice_platform.engine import HealthcareVoiceAssistant


def main():
    parser = argparse.ArgumentParser(description="Healthcare Voice Assistant")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Platform config path",
    )
    parser.add_argument(
        "--healthcare-config",
        default="configs/healthcare/clinic.yaml",
        help="Healthcare config path",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run in text-only mode (no audio)",
    )
    args = parser.parse_args()
    
    if args.text_only:
        # Text-only mode for testing
        run_text_mode(args.healthcare_config)
    else:
        # Full voice mode
        assistant = HealthcareVoiceAssistant(
            config_path=args.config,
            healthcare_config_path=args.healthcare_config,
        )
        assistant.run()


def run_text_mode(healthcare_config_path: str):
    """Run in text-only mode for testing without audio."""
    from voice_platform.healthcare import (
        HealthcareConversationAgent,
        load_healthcare_config,
    )
    
    config = load_healthcare_config(healthcare_config_path)
    
    agent = HealthcareConversationAgent(
        clinic_name=config.clinic.name,
        require_verification=False,  # Skip for text testing
        available_slots=config.available_slots,
    )
    
    print("\n" + "=" * 60)
    print(f"🏥 {config.clinic.name} - Text Mode")
    print("   Type 'quit' to exit")
    print("=" * 60 + "\n")
    
    # Start conversation
    response = agent.start()
    print(f"🤖 Agent: {response.message}\n")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            response = agent.process(user_input)
            print(f"🤖 Agent: {response.message}")
            
            if response.slots:
                print(f"   📋 Slots: {response.slots}")
            
            print()
            
            if response.ended:
                print("✅ Conversation complete!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
    
    agent.end()


if __name__ == "__main__":
    main()
