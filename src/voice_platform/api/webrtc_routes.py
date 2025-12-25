"""WebRTC signaling routes."""
import json
import asyncio
from aiohttp import web

from ..channels.webrtc import WebRTCChannel
from ..logging import get_logger

logger = get_logger("api.webrtc")


def setup_webrtc_routes(app, app_state):
    """Setup WebRTC routes on aiohttp app."""
    
    # Create WebRTC channel manager
    webrtc_channel = WebRTCChannel(app_state)
    app_state.webrtc_channel = webrtc_channel
    
    async def websocket_handler(request):
        """Handle WebRTC signaling over WebSocket."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        session_id = None
        session = None
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data['type'] == 'offer':
                        # Create new session
                        session_id, pc = await webrtc_channel.create_session()
                        session = webrtc_channel.sessions[session_id]
                        
                        # Store websocket for sending transcripts
                        session.ws = ws
                        
                        # Handle the offer
                        answer_sdp = await webrtc_channel.handle_offer(session_id, data['sdp'])
                        
                        await ws.send_json({
                            'type': 'answer',
                            'sdp': answer_sdp,
                            'session_id': session_id
                        })
                        
                        logger.info("webrtc_signaling_complete", session=session_id)
                        
                    elif data['type'] == 'ice' and session:
                        # Handle ICE candidate
                        if data.get('candidate'):
                            from aiortc import RTCIceCandidate
                            # ICE candidates are handled automatically by aiortc
                            pass
                            
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error("websocket_error", error=ws.exception())
                    
        except Exception as e:
            logger.error("webrtc_ws_error", error=str(e))
        finally:
            if session_id:
                await webrtc_channel.close_session(session_id)
            
        return ws
    
    async def index_handler(request):
        """Serve the demo page."""
        import os
        web_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'web')
        index_path = os.path.join(web_dir, 'index.html')
        
        if os.path.exists(index_path):
            return web.FileResponse(index_path)
        else:
            return web.Response(text="Demo page not found", status=404)
    
    # Add routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws/webrtc', websocket_handler)
    
    logger.info("webrtc_routes_configured")
