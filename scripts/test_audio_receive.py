"""Simple test server to debug audio receiving."""
import sys
sys.path.insert(0, 'src')

import asyncio
import base64
import json
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
    if from_sr == to_sr:
        return audio
    duration = len(audio) / from_sr
    new_length = int(duration * to_sr)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


@app.get("/", response_class=HTMLResponse)
async def index():
    return '''<!DOCTYPE html>
<html>
<head><title>Audio Test</title></head>
<body style="background:#1a1a2e;color:#fff;font-family:sans-serif;padding:20px;">
    <h1>Audio Capture Test</h1>
    <p id="status">Click Start to begin</p>
    <button id="startBtn" style="padding:20px 40px;font-size:18px;">Start Recording</button>
    <button id="stopBtn" style="padding:20px 40px;font-size:18px;" disabled>Stop</button>
    <div id="log" style="margin-top:20px;padding:10px;background:#333;height:300px;overflow-y:auto;font-family:monospace;"></div>
    
    <script>
        const status = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const log = document.getElementById('log');
        
        let ws = null;
        let audioContext = null;
        let mediaStream = null;
        let isRecording = false;
        
        function addLog(msg) {
            const div = document.createElement('div');
            div.textContent = new Date().toLocaleTimeString() + ': ' + msg;
            log.appendChild(div);
            log.scrollTop = log.scrollHeight;
        }
        
        startBtn.onclick = async () => {
            try {
                // Connect WebSocket
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${location.host}/ws`);
                
                ws.onopen = () => {
                    addLog('WebSocket connected');
                    status.textContent = 'Connected';
                };
                
                ws.onmessage = (e) => {
                    const data = JSON.parse(e.data);
                    addLog('Server: ' + JSON.stringify(data));
                };
                
                ws.onerror = (e) => addLog('WS Error: ' + e);
                ws.onclose = () => addLog('WS Closed');
                
                // Get microphone
                addLog('Requesting microphone...');
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                addLog('Microphone access granted');
                
                // Create audio context with DEFAULT sample rate
                audioContext = new AudioContext();
                const sampleRate = audioContext.sampleRate;
                addLog('AudioContext created, sampleRate: ' + sampleRate);
                
                // Send sample rate to server
                ws.send(JSON.stringify({ type: 'config', sampleRate: sampleRate }));
                
                const source = audioContext.createMediaStreamSource(mediaStream);
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (!isRecording) return;
                    
                    const inputData = e.inputBuffer.getChannelData(0);
                    const float32Array = new Float32Array(inputData);
                    
                    // Calculate energy
                    let sum = 0;
                    for (let i = 0; i < float32Array.length; i++) {
                        sum += float32Array[i] * float32Array[i];
                    }
                    const energy = Math.sqrt(sum / float32Array.length);
                    
                    // Convert Float32Array to base64
                    const bytes = new Uint8Array(float32Array.buffer);
                    let binary = '';
                    for (let i = 0; i < bytes.length; i++) {
                        binary += String.fromCharCode(bytes[i]);
                    }
                    const audio_b64 = btoa(binary);
                    
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ 
                            type: 'audio', 
                            audio: audio_b64
                        }));
                    }
                    
                    // Log occasionally
                    if (Math.random() < 0.05) {
                        addLog(`Energy: ${energy.toFixed(4)}`);
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isRecording = true;
                startBtn.disabled = true;
                stopBtn.disabled = false;
                status.textContent = 'Recording... Speak now!';
                addLog('Recording started');
                
            } catch (err) {
                addLog('Error: ' + err.message);
                status.textContent = 'Error: ' + err.message;
            }
        };
        
        stopBtn.onclick = () => {
            isRecording = false;
            if (mediaStream) {
                mediaStream.getTracks().forEach(t => t.stop());
            }
            if (audioContext) {
                audioContext.close();
            }
            if (ws) {
                ws.close();
            }
            startBtn.disabled = false;
            stopBtn.disabled = true;
            status.textContent = 'Stopped';
            addLog('Recording stopped');
        };
    </script>
</body>
</html>'''


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    chunk_count = 0
    client_sample_rate = 48000  # Default, will be updated by config
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "config":
                client_sample_rate = data.get("sampleRate", 48000)
                print(f"Client sample rate: {client_sample_rate}")
                await websocket.send_json({"type": "config_ack", "sampleRate": client_sample_rate})
            
            elif data.get("type") == "audio":
                audio_b64 = data.get("audio", "")
                
                # Decode
                audio_bytes = base64.b64decode(audio_b64)
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Resample to 16kHz
                audio_16k = resample(audio, client_sample_rate, 16000)
                
                chunk_count += 1
                energy = np.sqrt(np.mean(audio_16k ** 2))
                
                # Log every 10th chunk
                if chunk_count % 10 == 0:
                    print(f"Chunk {chunk_count}: {len(audio)} -> {len(audio_16k)} samples (16kHz), energy={energy:.4f}")
                    
                    await websocket.send_json({
                        "type": "ack",
                        "chunks": chunk_count,
                        "samples": len(audio_16k),
                        "energy": float(energy)
                    })
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("WebSocket disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
