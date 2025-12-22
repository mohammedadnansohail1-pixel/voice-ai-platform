"""Run the Voice AI server."""
import sys
sys.path.insert(0, 'src')

import uvicorn
from voice_platform.api.server import create_app
from voice_platform.logging import setup_logging
from voice_platform.core.config import LoggingConfig

if __name__ == "__main__":
    setup_logging(LoggingConfig(level="INFO", format="console"))
    
    app = create_app(flow_path="configs/flows/appointment.yaml")
    
    print("\n" + "="*50)
    print("🎙️  Voice AI Platform")
    print("="*50)
    print("Open in browser: http://localhost:8000")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
