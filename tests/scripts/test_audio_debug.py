"""Debug audio receiving."""
import sys
sys.path.insert(0, 'src')

import asyncio
import base64
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index():
    return '''<!DOCTYPE html>
<html>
<head><title>Audio Debug</title></head>
<body style="background:#1a1a2e;color:#fff;font-family:sans-serif;padding:20px;">
    <h1>Audio Debug</h1>
    <p>Status: <span id="status">Not started</span></p>
    <p>Chunks sent: <span id="chunks">0</span></p>
    <p>Last energy: <span id="energy">0</span></p>
    <button id="startBtn" style="padding:20px;font-size:18px;">Start Recording</button>
    <button id="stopBtn" style="padding:20px;font-size:18px;" disabled>Stop</button>
    <div id="log" style="margin-top:20px;background:#333;padding:10px;height:300px;overflow-y:auto;font-family:monospace;font-size:12px;"></div>
    
    <script>
        const status = document.getElementById('status');
        const chunksEl = document.getElementById('chunks');
        const energyEl = document.getElementById('energy');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const log = document.getElementById('log');
        
        let ws, audioContext, mediaStream, processor, isRecording = false, chunks = 0;
        
        function addLog(msg) {
            const div = document.createElement('div');
            div.textContent = new Date().toLocaleTimeString() + ': ' + msg;
            log.appendChild(div);
            log.scrollTop = log.scrollHeight;
            console.log(msg);
        }
        
        startBtn.onclick = async () => {
            try {
                addLog('Connecting WebSocket...');
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${location.host}/ws`);
                
                ws.onopen = () => addLog('WebSocket connected');
                ws.onmessage = (e) => {
                    const data = JSON.parse(e.data);
                    addLog('Server: ' + JSON.stringify(data));
                };
                ws.onerror = (e) => addLog('WS Error');
                ws.onclose = () => addLog('WS Closed');
                
                addLog('Requesting microphone...');
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: true, noiseSuppression: true }
                });
                addLog('Microphone granted, tracks: ' + mediaStream.getAudioTracks().length);
                
                audioContext = new AudioContext();
                addLog('AudioContext sampleRate: ' + audioContext.sampleRate);
                
                if (audioContext.state === 'suspended') {
                    await audioContext.resume();
                    addLog('AudioContext resumed');
                }
                
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (!isRecording) return;
                    
                    const data = e.inputBuffer.getChannelData(0);
                    
                    // Calculate energy
                    let sum = 0;
                    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
                    const energy = Math.sqrt(sum / data.length);
                    energyEl.textContent = energy.toFixed(4);
                    
                    // Send to server
                    const bytes = new Uint8Array(new Float32Array(data).buffer);
                    let b = '';
                    for (let i = 0; i < bytes.length; i++) b += String.fromCharCode(bytes[i]);
                    
                    if (ws && ws.readyState === 1) {
                        ws.send(JSON.stringify({ type: 'audio', audio: btoa(b), energy: energy }));
                        chunks++;
                        chunksEl.textContent = chunks;
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isRecording = true;
                status.textContent = 'Recording';
                startBtn.disabled = true;
                stopBtn.disabled = false;
                addLog('Recording started');
                
            } catch (err) {
                addLog('Error: ' + err.message);
                status.textContent = 'Error: ' + err.message;
            }
        };
        
        stopBtn.onclick = () => {
            isRecording = false;
            if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
            if (audioContext) audioContext.close();
            if (ws) ws.close();
            status.textContent = 'Stopped';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            addLog('Stopped');
        };
    </script>
</body>
</html>'''

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Connected")
    
    chunks = 0
    speech_chunks = 0
    is_speaking = False
    
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "audio":
                audio_b64 = data.get("audio", "")
                client_energy = data.get("energy", 0)
                
                audio_bytes = base64.b64decode(audio_b64)
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
                server_energy = np.sqrt(np.mean(audio ** 2))
                
                chunks += 1
                
                if server_energy > 0.015:
                    if not is_speaking:
                        is_speaking = True
                        print(f"\n>>> SPEECH START (chunk {chunks})")
                    speech_chunks += 1
                elif is_speaking:
                    print(f"<<< SPEECH END after {speech_chunks} chunks")
                    is_speaking = False
                    speech_chunks = 0
                
                if chunks % 20 == 0:
                    print(f"Chunk {chunks}: client_energy={client_energy:.4f}, server_energy={server_energy:.4f}, speaking={is_speaking}")
                    await websocket.send_json({"chunks": chunks, "energy": server_energy})
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
