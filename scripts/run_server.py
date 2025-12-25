#!/usr/bin/env python3
"""Run the voice platform server with AudioSocket support."""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn


async def start_audiosocket_server(app_state, config, host="0.0.0.0", port=9000):
    """Start general AudioSocket server for Asterisk."""
    from voice_platform.channels.asterisk import handle_asterisk_connection
    from voice_platform.logging import get_logger
    
    logger = get_logger("audiosocket")

    async def client_handler(reader, writer):
        await handle_asterisk_connection(reader, writer, app_state, config)

    server = await asyncio.start_server(client_handler, host, port)
    logger.info("audiosocket_server_started", host=host, port=port, mode="general")
    
    async with server:
        await server.serve_forever()


async def start_healthcare_audiosocket_server(app_state, config, host="0.0.0.0", port=9001):
    """Start healthcare AudioSocket server for Asterisk."""
    from voice_platform.channels.asterisk_healthcare import handle_healthcare_connection
    from voice_platform.logging import get_logger
    
    logger = get_logger("audiosocket_healthcare")

    async def client_handler(reader, writer):
        await handle_healthcare_connection(reader, writer, app_state, config)

    server = await asyncio.start_server(client_handler, host, port)
    logger.info("audiosocket_server_started", host=host, port=port, mode="healthcare")
    
    async with server:
        await server.serve_forever()


async def main():
    from voice_platform.api.server import create_app, VoicePlatformApp
    from voice_platform.core.config import load_config
    from voice_platform.logging import setup_logging, get_logger

    logger = get_logger("main")

    config_path = "configs/base.yaml"
    config = load_config(config_path)
    setup_logging(config.logging)

    # Create app state and load models
    app_state = VoicePlatformApp(config)
    app_state.load_models()

    # Start AudioSocket servers
    general_task = asyncio.create_task(
        start_audiosocket_server(app_state, config, port=9000)
    )
    healthcare_task = asyncio.create_task(
        start_healthcare_audiosocket_server(app_state, config, port=9001)
    )

    logger.info("all_audiosocket_servers_started", general=9000, healthcare=9001)

    # Start HTTP/WebSocket server
    app = create_app(config_path)
    app.state.platform = app_state  # Share loaded models

    uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(uvicorn_config)
    
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
