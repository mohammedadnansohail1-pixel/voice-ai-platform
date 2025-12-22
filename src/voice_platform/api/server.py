"""FastAPI WebSocket server for voice AI."""
import asyncio
import base64
import uuid
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np

from ..flows import FlowEngine, load_flow
from ..logging import get_logger

logger = get_logger("api.server")

_asr = None
_tts = None


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    duration = len(audio) / from_sr
    new_length = int(duration * to_sr)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def get_asr():
    global _asr
    if _asr is None:
        from ..pipeline.asr_whisper import WhisperASR
        _asr = WhisperASR(model="large-v3-turbo", device="cuda", compute_type="float16")
        _asr.load()
    return _asr


def get_tts():
    global _tts
    if _tts is None:
        from ..pipeline.tts_kokoro import KokoroTTS
        _tts = KokoroTTS(voice="af_heart", device="cuda")
        _tts.load()
    return _tts


def create_app(flow_path: Optional[str] = None, preload_models: bool = True) -> FastAPI:
    app = FastAPI(title="Voice AI Platform", version="1.0.0")
    
    flow = None
    if flow_path and Path(flow_path).exists():
        flow = load_flow(flow_path)
        logger.info("flow_loaded", flow=flow.name)
    
    if preload_models:
        logger.info("preloading_models")
        get_asr()
        get_tts()
        logger.info("models_ready")
    
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return get_browser_client_html()
    
    @app.get("/health")
    async def health():
        return {"status": "ok", "flow": flow.name if flow else None}
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        
        session_id = str(uuid.uuid4())[:8]
        logger.info("websocket_connected", session_id=session_id)
        
        audio_buffer = []
        is_speaking = False
        silence_start = None
        client_sample_rate = 48000
        bot_speaking = False  # Server-side mute flag
        bot_speaking_until = 0  # Timestamp when bot finishes speaking
        
        asr = get_asr()
        tts = get_tts()
        
        engine = None
        if flow:
            engine = FlowEngine(flow)
            response = engine.start()
            if response.message:
                await websocket.send_json({
                    "type": "message",
                    "text": response.message,
                    "state": response.current_state,
                })
                duration = await synthesize_and_send(websocket, tts, response.message)
                bot_speaking_until = time.time() + duration + 0.5  # Add buffer
        
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    msg_type = data.get("type")
                    
                    if msg_type == "config":
                        client_sample_rate = data.get("sampleRate", 48000)
                        logger.info("client_config", sample_rate=client_sample_rate)
                        await websocket.send_json({"type": "config_ack"})
                    
                    elif msg_type == "audio":
                        # Server-side mute: ignore audio while bot is speaking
                        if time.time() < bot_speaking_until:
                            continue
                        
                        audio_b64 = data.get("audio", "")
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                        audio_16k = resample(audio_chunk, client_sample_rate, 16000)
                        
                        energy = np.sqrt(np.mean(audio_16k ** 2))
                        
                        if energy > 0.015:
                            if not is_speaking:
                                is_speaking = True
                                audio_buffer = []
                                logger.info("speech_start", session_id=session_id)
                            audio_buffer.append(audio_16k)
                            silence_start = None
                        elif is_speaking:
                            audio_buffer.append(audio_16k)
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > 0.6:
                                is_speaking = False
                                logger.info("speech_end", session_id=session_id, chunks=len(audio_buffer))
                                
                                if audio_buffer:
                                    full_audio = np.concatenate(audio_buffer)
                                    audio_buffer = []
                                    
                                    t0 = time.time()
                                    loop = asyncio.get_event_loop()
                                    result = await loop.run_in_executor(None, asr.transcribe, full_audio, 16000)
                                    t1 = time.time()
                                    
                                    user_text = result.text.strip()
                                    logger.info("transcribed", session_id=session_id, text=user_text, time=f"{t1-t0:.2f}s")
                                    
                                    if user_text and engine:
                                        await websocket.send_json({"type": "transcription", "text": user_text})
                                        
                                        response = engine.process_input(user_text)
                                        if response.action_request:
                                            response = engine.execute_action_result(success=True)
                                        
                                        if response.message:
                                            await websocket.send_json({
                                                "type": "message",
                                                "text": response.message,
                                                "state": response.current_state,
                                                "slots": response.slots,
                                                "ended": response.ended,
                                            })
                                            duration = await synthesize_and_send(websocket, tts, response.message)
                                            bot_speaking_until = time.time() + duration + 0.5
                    
                    elif msg_type == "mic_ready":
                        # Client signals it's ready to record
                        logger.info("mic_ready", session_id=session_id)
                    
                    elif msg_type == "text" and engine:
                        user_text = data.get("text", "")
                        response = engine.process_input(user_text)
                        if response.action_request:
                            response = engine.execute_action_result(success=True)
                        
                        await websocket.send_json({
                            "type": "message",
                            "text": response.message or "",
                            "state": response.current_state,
                            "slots": response.slots,
                            "ended": response.ended,
                        })
                        if response.message:
                            duration = await synthesize_and_send(websocket, tts, response.message)
                            bot_speaking_until = time.time() + duration + 0.5
                    
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error("error", error=str(e))
                    import traceback
                    traceback.print_exc()
                    break
        finally:
            logger.info("disconnected", session_id=session_id)
    
    return app


async def synthesize_and_send(websocket: WebSocket, tts, text: str) -> float:
    """Synthesize and send audio. Returns duration in seconds."""
    try:
        await websocket.send_json({"type": "mic_mute"})
        
        t0 = time.time()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, tts.synthesize, text)
        t1 = time.time()
        
        logger.info("tts_done", time=f"{t1-t0:.2f}s", duration=f"{result.duration_seconds:.2f}s")
        
        audio_b64 = base64.b64encode(result.audio.tobytes()).decode('utf-8')
        await websocket.send_json({
            "type": "audio",
            "audio": audio_b64,
            "sample_rate": result.sample_rate,
            "duration": result.duration_seconds,
        })
        return result.duration_seconds
    except Exception as e:
        logger.error("tts_error", error=str(e))
        return 0.0


def get_browser_client_html() -> str:
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice AI Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh; color: #fff; padding: 20px;
        }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
        .status { display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 20px; }
        .status-dot { width: 12px; height: 12px; border-radius: 50%; background: #ff4444; }
        .status-dot.connected { background: #44ff44; }
        .status-dot.recording { background: #ff8844; animation: pulse 1s infinite; }
        .status-dot.speaking { background: #4488ff; animation: pulse 0.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .chat-container {
            background: rgba(255,255,255,0.05); border-radius: 12px;
            padding: 20px; height: 350px; overflow-y: auto; margin-bottom: 20px;
        }
        .message { margin-bottom: 15px; padding: 12px 16px; border-radius: 12px; max-width: 85%; }
        .message.bot { background: #0066cc; margin-right: auto; }
        .message.user { background: #444; margin-left: auto; }
        .controls { display: flex; gap: 10px; justify-content: center; margin-bottom: 20px; }
        button { padding: 15px 30px; border-radius: 12px; border: none; font-size: 1rem; cursor: pointer; }
        .mic-btn { background: #0066cc; color: #fff; min-width: 200px; }
        .mic-btn.recording { background: #cc3300; }
        .mic-btn.muted { background: #555; }
        .mic-btn:disabled { background: #333; cursor: not-allowed; }
        .input-container { display: flex; gap: 10px; }
        input[type="text"] {
            flex: 1; padding: 15px; border-radius: 12px; border: none;
            background: rgba(255,255,255,0.1); color: #fff; font-size: 1rem;
        }
        .send-btn { background: #0066cc; color: #fff; }
        .slots { margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 12px; }
        .slots h3 { margin-bottom: 10px; font-size: 0.9rem; color: #888; }
        .slot-item { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Voice AI</h1>
        <p class="subtitle">Appointment Scheduling</p>
        <div class="status">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Connecting...</span>
        </div>
        <div class="chat-container" id="chatContainer"></div>
        <div class="controls">
            <button class="mic-btn" id="micBtn" disabled>🎤 Click to Talk</button>
        </div>
        <div class="input-container">
            <input type="text" id="userInput" placeholder="Or type..." disabled>
            <button class="send-btn" id="sendBtn" disabled>Send</button>
        </div>
        <div class="slots" id="slotsContainer" style="display: none;">
            <h3>Collected Info</h3>
            <div id="slotsList"></div>
        </div>
    </div>
    <script>
        const chatContainer = document.getElementById('chatContainer');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const micBtn = document.getElementById('micBtn');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const slotsContainer = document.getElementById('slotsContainer');
        const slotsList = document.getElementById('slotsList');
        
        let ws, audioContext, mediaStream, processor;
        let isRecording = false;
        let isMuted = true;  // Start muted until first audio plays
        let clientSampleRate = 48000;
        let audioQueue = [], isPlaying = false;
        
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            
            ws.onopen = async () => {
                statusDot.classList.add('connected');
                statusText.textContent = 'Initializing...';
                await initAudio();
            };
            ws.onclose = () => {
                statusDot.className = 'status-dot';
                statusText.textContent = 'Disconnected';
                micBtn.disabled = userInput.disabled = sendBtn.disabled = true;
                setTimeout(connect, 2000);
            };
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                
                if (data.type === 'config_ack') {
                    statusText.textContent = 'Waiting for greeting...';
                    userInput.disabled = sendBtn.disabled = false;
                }
                
                if (data.type === 'mic_mute') {
                    isMuted = true;
                    isRecording = false;
                    micBtn.classList.remove('recording');
                    micBtn.classList.add('muted');
                    micBtn.textContent = '🔇 Bot Speaking...';
                    micBtn.disabled = true;
                    statusDot.className = 'status-dot speaking';
                    statusText.textContent = 'Bot speaking...';
                }
                
                if (data.type === 'message' && data.text) {
                    addMessage(data.text, 'bot');
                    if (data.slots && Object.keys(data.slots).length) updateSlots(data.slots);
                    if (data.ended) {
                        micBtn.disabled = userInput.disabled = sendBtn.disabled = true;
                        statusText.textContent = 'Conversation ended';
                    }
                }
                
                if (data.type === 'transcription') {
                    addMessage(data.text, 'user');
                }
                
                if (data.type === 'audio') {
                    playAudio(data.audio, data.sample_rate, data.duration || 3);
                }
            };
        }
        
        async function initAudio() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } 
                });
                audioContext = new AudioContext();
                clientSampleRate = audioContext.sampleRate;
                ws.send(JSON.stringify({ type: 'config', sampleRate: clientSampleRate }));
                
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                processor.onaudioprocess = (e) => {
                    // Only send if actively recording and not muted
                    if (!isRecording || isMuted || !ws || ws.readyState !== 1) return;
                    
                    const data = e.inputBuffer.getChannelData(0);
                    const bytes = new Uint8Array(new Float32Array(data).buffer);
                    let b = ''; 
                    for (let i = 0; i < bytes.length; i++) b += String.fromCharCode(bytes[i]);
                    ws.send(JSON.stringify({ type: 'audio', audio: btoa(b) }));
                };
                source.connect(processor);
                processor.connect(audioContext.destination);
            } catch (err) { 
                statusText.textContent = 'Mic error: ' + err.message; 
            }
        }
        
        function playAudio(b64, sr, expectedDuration) {
            audioQueue.push({ b64, sr, expectedDuration });
            if (!isPlaying) playNext();
        }
        
        async function playNext() {
            if (!audioQueue.length) { 
                isPlaying = false;
                // Audio finished - enable mic after delay
                setTimeout(() => {
                    isMuted = false;
                    micBtn.classList.remove('muted');
                    micBtn.textContent = '🎤 Click to Talk';
                    micBtn.disabled = false;
                    statusDot.className = 'status-dot connected';
                    statusText.textContent = 'Your turn - Click mic to speak';
                    ws.send(JSON.stringify({ type: 'mic_ready' }));
                }, 500);  // 500ms buffer after audio ends
                return; 
            }
            isPlaying = true;
            const { b64, sr } = audioQueue.shift();
            
            try {
                const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
                const floats = new Float32Array(bytes.buffer);
                const ctx = new AudioContext({ sampleRate: sr });
                const buf = ctx.createBuffer(1, floats.length, sr);
                buf.getChannelData(0).set(floats);
                const src = ctx.createBufferSource();
                src.buffer = buf;
                src.connect(ctx.destination);
                src.onended = () => { ctx.close(); playNext(); };
                src.start();
            } catch (e) { 
                console.error('Playback error:', e);
                playNext(); 
            }
        }
        
        function addMessage(text, sender) {
            const div = document.createElement('div');
            div.className = 'message ' + sender;
            div.textContent = text;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function updateSlots(slots) {
            slotsContainer.style.display = 'block';
            slotsList.innerHTML = Object.entries(slots)
                .map(([k,v]) => `<div class="slot-item"><span>${k}</span><span>${v}</span></div>`)
                .join('');
        }
        
        function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;
            addMessage(text, 'user');
            ws.send(JSON.stringify({ type: 'text', text }));
            userInput.value = '';
        }
        
        micBtn.onclick = () => {
            if (isMuted) return;
            
            if (audioContext?.state === 'suspended') audioContext.resume();
            
            isRecording = !isRecording;
            
            if (isRecording) {
                micBtn.classList.add('recording');
                micBtn.textContent = '🔴 Click to Stop';
                statusDot.className = 'status-dot recording';
                statusText.textContent = 'Listening... Click to stop';
            } else {
                micBtn.classList.remove('recording');
                micBtn.textContent = '🎤 Click to Talk';
                statusDot.className = 'status-dot connected';
                statusText.textContent = 'Processing...';
            }
        };
        
        sendBtn.onclick = sendMessage;
        userInput.onkeypress = (e) => { if (e.key === 'Enter') sendMessage(); };
        connect();
    </script>
</body>
</html>'''
