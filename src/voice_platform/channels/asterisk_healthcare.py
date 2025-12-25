"""Healthcare Asterisk AudioSocket handler with Tool-Calling Agent."""
import asyncio

from .asterisk import AsteriskAudioSocket
from ..core.config import Config
from ..audio.accumulator import SpeechAccumulator
from ..agent import ToolCallingAgent
from ..logging import get_logger

logger = get_logger("channels.asterisk_healthcare")


async def handle_healthcare_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    app_state,
    config: Config
):
    """
    Handle Asterisk AudioSocket connection for healthcare scheduling.
    
    Uses ToolCallingAgent with:
    - Rule-based slot extraction (no LLM hallucinations)
    - LLM for natural response generation only
    """
    channel = AsteriskAudioSocket(reader, writer, config)
    await channel.accept()
    
    if not channel.is_connected:
        return

    # Load healthcare config
    try:
        from ..healthcare.config import load_healthcare_config
        healthcare_config = load_healthcare_config("configs/healthcare/clinic.yaml")
        clinic_name = healthcare_config.clinic.name
        available_slots = healthcare_config.available_slots
    except Exception as e:
        logger.warning("healthcare_config_fallback", error=str(e))
        clinic_name = "Sunrise Medical"
        available_slots = None

    # Initialize tool-calling agent
    agent = ToolCallingAgent(
        llm=app_state.llm,
        clinic_name=clinic_name,
        available_slots=available_slots,
    )
    
    # Speech accumulator
    accumulator = SpeechAccumulator(config.vad)
    
    # Audit logging
    app_state.audit.session_start(
        channel.session.session_id, 
        channel="asterisk_healthcare_tool_calling"
    )

    try:
        # Send greeting
        greeting = agent.start()
        logger.info("healthcare_greeting", message=greeting[:60])
        
        tts_result = app_state.tts.synthesize(greeting)
        await channel.send_audio(tts_result.audio_data, tts_result.sample_rate)

        async for audio_chunk in channel.receive_audio():
            vad_result = app_state.vad.process_chunk(audio_chunk)
            segment = accumulator.process(audio_chunk, vad_result)
            
            if segment:
                # Transcribe
                transcript = app_state.asr.transcribe(segment.audio)
                
                if not transcript.text.strip():
                    continue
                
                user_text = transcript.text.strip()
                
                # Log with confidence for debugging
                asr_confidence = 0.75
                if transcript.segments:
                    confidences = [s.confidence for s in transcript.segments if s.confidence]
                    if confidences:
                        avg_logprob = sum(confidences) / len(confidences)
                        asr_confidence = min(1.0, max(0.0, 1.0 + (avg_logprob * 0.5)))
                
                logger.info(
                    "healthcare_user",
                    text=user_text,
                    confidence=f"{asr_confidence:.2f}",
                )
                
                # Process through agent directly
                response = agent.process(user_text)
                ctx = agent.get_context()
                
                logger.info(
                    "healthcare_agent_response",
                    response=response.text[:80],
                    state=ctx.get("state"),
                    reason=ctx.get("visit_reason"),
                    day=ctx.get("preferred_day"),
                    time=ctx.get("preferred_time"),
                )
                
                # Synthesize and send response
                tts_result = app_state.tts.synthesize(response.text)
                await channel.send_audio(tts_result.audio_data, tts_result.sample_rate)
                
                # Check if conversation ended
                if response.ended:
                    if response.booking:
                        logger.info("healthcare_booking_complete", booking=response.booking)
                    elif response.transfer:
                        logger.info("healthcare_transfer_requested")
                    break

    except Exception as e:
        logger.error("healthcare_error", error=str(e), exc_info=True)
        
    finally:
        app_state.audit.session_end(
            channel.session.session_id,
            duration_s=channel.session.duration_s
        )
        await channel.close()
