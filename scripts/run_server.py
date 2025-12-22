#!/usr/bin/env python3
"""Run the voice platform API server."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import uvicorn
from voice_platform.api import create_app

app = create_app("configs/base.yaml")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
