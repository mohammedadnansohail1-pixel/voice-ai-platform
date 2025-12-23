"""Twilio telephony webhook routes."""
from fastapi import APIRouter, Request, Response, WebSocket
from fastapi.responses import PlainTextResponse

from ...logging import get_logger

logger = get_logger("api.telephony")

router = APIRouter(prefix="/telephony", tags=["telephony"])


@router.post("/voice/incoming")
async def incoming_call(request: Request) -> Response:
    """
    Handle incoming Twilio voice call.
    
    Returns TwiML to connect to Media Streams WebSocket.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    from_number = form.get("From", "unknown")
    to_number = form.get("To", "unknown")
    
    logger.info(
        "incoming_call",
        call_sid=call_sid,
        from_number=from_number,
        to_number=to_number,
    )
    
    # Get the host for WebSocket URL
    host = request.headers.get("host", "localhost:8000")
    ws_url = f"wss://{host}/telephony/media-stream"
    
    # Return TwiML to start media stream
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Welcome to the voice assistant. Please wait while I connect you.</Say>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="callSid" value="{call_sid}"/>
        </Stream>
    </Connect>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


@router.post("/voice/status")
async def call_status(request: Request) -> Response:
    """Handle call status callbacks."""
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    call_status = form.get("CallStatus", "unknown")
    
    logger.info("call_status", call_sid=call_sid, status=call_status)
    
    return PlainTextResponse("OK")


@router.post("/voice/fallback")
async def voice_fallback(request: Request) -> Response:
    """Fallback handler for errors."""
    form = await request.form()
    error_code = form.get("ErrorCode", "unknown")
    error_message = form.get("ErrorMessage", "unknown")
    
    logger.error("twilio_error", code=error_code, message=error_message)
    
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I'm sorry, we're experiencing technical difficulties. Please try again later.</Say>
    <Hangup/>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")
