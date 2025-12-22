"""Simple audio test."""
import asyncio
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

HTML_PAGE = """<!DOCTYPE html>
<html>
<head><title>Audio Test</title></head>
<body style="background:#222;color:#fff;font-family:sans-serif;padding:20px;">
    <h1>Audio Test</h1>
    <p>Status: <span id="statusEl" style="color:#0f0;">Connecting...</span></p>
    <p>Chunks: <span id="chunksEl">0</span> | Energy: <span id="energyEl">0</span></p>
    <button id="btn" style="padding:30px 60px;font-size:24px;cursor:pointer;">Hold to Talk</button>
    <pre id="logEl" style="background:#333;padding:10px;height:400px;overflow-y:auto;font-size:11px;"></pre>
    
    <script>
        var statusEl = document.getElementById('statusEl');
        var chunksEl = document.getElementById('chunksEl');
        var energyEl = document.getElementById('energyEl');
        var btn = document.getElementById('btn');
        var logEl = document.getElementById('logEl');
        
        var ws, audioContext, mediaStream, processor;
        var isRecording = false, chunkCount = 0;
        
        function addLog(msg) {
            logEl.textContent += new Date().toLocaleTimeString() + ': ' + msg + '\\n';
            logEl.scrollTop = logEl.scrollHeight;
        }
        
        async function init() {
            try {
                var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(protocol + '//' + location.host + '/ws');
                
                ws.onopen = function() {
                    addLog('WebSocket connected');
                    statusEl.textContent = 'Connected';
                    statusEl.style.color = '#0f0';
                };
                ws.onmessage = function(e) { addLog('Server: ' + e.data); };
                ws.onerror = function(e) { addLog('WS Error'); };
                ws.onclose = function(e) {
                    addLog('WS Closed: code=' + e.code);
                    statusEl.textContent = 'Disconnected';
                    statusEl.style.color = '#f00';
                };
                
                addLog('Requesting mic...');
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                addLog('Mic granted');
                
                audioContext = new AudioContext();
                addLog('AudioContext: ' + audioContext.sampleRate + 'Hz');
                
                var source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = function(e) {
                    if (!isRecording || !ws || ws.readyState !== 1) return;
                    
                    var data = e.inputBuffer.getChannelData(0);
                    var sum = 0;
                    for (var i = 0; i < data.length; i++) sum += data[i] * data[i];
                    var energy = Math.sqrt(sum / data.length);
                    energyEl.textContent = energy.toFixed(4);
                    
                    var floats = new Float32Array(data);
                    var bytes = new Uint8Array(floats.buffer);
                    var binary = '';
                    for (var i = 0; i < bytes.length; i++) {
                        binary += String.fromCharCode(bytes[i]);
                    }
                    
                    ws.send(JSON.stringify({ type: 'audio', audio: btoa(binary) }));
                    chunkCount++;
                    chunksEl.textContent = chunkCount;
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                addLog('Ready');
                
            } catch (err) {
                addLog('Error: ' + err.message);
            }
        }
        
        btn.onmousedown = async function() {
            if (audioContext && audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            isRecording = true;
            btn.textContent = 'Recording...';
            btn.style.background = '#c00';
        };
        
        btn.onmouseup = function() {
            isRecording = false;
            btn.textContent = 'Hold to Talk';
            btn.style.background = '';
        };
        
        btn.onmouseleave = function() {
            if (isRecording) {
                isRecording = false;
                btn.textContent = 'Hold to Talk';
                btn.style.background = '';
            }
        };
        
        init();
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    chunk_count = 0
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                audio_b64 = data.get("audio", "")
                audio_bytes = base64.b64decode(audio_b64)
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
                energy = float(np.sqrt(np.mean(audio ** 2)))
                
                chunk_count += 1
                
                if chunk_count % 10 == 0:
                    print(f"Chunk {chunk_count}: energy={energy:.4f}")
                    await websocket.send_json({"chunks": chunk_count, "energy": round(energy, 4)})
                        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"Total chunks: {chunk_count}")

if __name__ == "__main__":
    print("Server: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
