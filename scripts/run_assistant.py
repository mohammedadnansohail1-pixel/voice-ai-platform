#!/usr/bin/env python3
"""Run the voice assistant."""
import sys
sys.path.insert(0, "src")

from voice_platform.engine import VoiceAssistant


def main():
    assistant = VoiceAssistant(config_path="configs/base.yaml")
    assistant.run()


if __name__ == "__main__":
    main()
