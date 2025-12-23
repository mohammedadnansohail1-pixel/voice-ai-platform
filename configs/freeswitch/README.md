# FreeSWITCH Integration

## Quick Start with Docker
```bash
# Run FreeSWITCH
docker run -d --name freeswitch \
  --network host \
  safarov/freeswitch

# Or use docker-compose (see below)
```

## Dialplan Configuration

Add to `/etc/freeswitch/dialplan/default.xml`:
```xml
<extension name="voice-ai">
  <condition field="destination_number" expression="^(1000)$">
    <action application="answer"/>
    <action application="set" data="RECORD_STEREO=true"/>
    <action application="set" data="media_bug_answer_req=true"/>
    
    <!-- Connect to Voice AI Platform via WebSocket -->
    <action application="lua" data="stream_to_websocket.lua"/>
  </condition>
</extension>
```

## Lua Script for WebSocket Streaming

Save as `/usr/share/freeswitch/scripts/stream_to_websocket.lua`:
```lua
-- stream_to_websocket.lua
local ws_url = "ws://YOUR_SERVER:8000/freeswitch/media-stream"

session:answer()
session:sleep(500)

-- Get call info
local uuid = session:getVariable("uuid")
local caller_id = session:getVariable("caller_id_number")
local destination = session:getVariable("destination_number")

-- Connect to WebSocket
local socket = freeswitch.Socket(ws_url)
socket:connect()

-- Send connect event
socket:send(string.format([[
  {"event": "connect", "uuid": "%s", "caller_id_number": "%s", "destination_number": "%s", "sample_rate": 16000}
]], uuid, caller_id, destination))

-- Stream audio bidirectionally
session:streamFile("silence_stream://0", "", function(s, type, data)
    if type == "audio" then
        socket:send_bytes(data)
    end
    
    -- Receive audio from AI and play
    local ai_audio = socket:recv_bytes(320)
    if ai_audio then
        return ai_audio
    end
end)

socket:close()
```

## Using mod_audio_stream (Alternative)

If you have `mod_audio_stream` compiled:
```xml
<extension name="voice-ai-stream">
  <condition field="destination_number" expression="^(1001)$">
    <action application="answer"/>
    <action application="audio_stream" data="ws://YOUR_SERVER:8000/freeswitch/media-stream start"/>
    <action application="park"/>
  </condition>
</extension>
```

## SIP Trunk Configuration

Add to `/etc/freeswitch/sip_profiles/external/voipms.xml`:
```xml
<gateway name="voipms">
  <param name="username" value="YOUR_VOIPMS_USERNAME"/>
  <param name="password" value="YOUR_VOIPMS_PASSWORD"/>
  <param name="realm" value="YOUR_SERVER.voip.ms"/>
  <param name="proxy" value="YOUR_SERVER.voip.ms"/>
  <param name="register" value="true"/>
</gateway>
```

## Route Incoming Calls

Add to dialplan:
```xml
<extension name="inbound-to-ai">
  <condition field="context" expression="public"/>
  <condition field="destination_number" expression="^(\+1NXXNXXXXXX)$">
    <action application="transfer" data="1000 XML default"/>
  </condition>
</extension>
```

## Testing

1. Start FreeSWITCH: `docker-compose up -d freeswitch`
2. Start Voice AI: `python scripts/run_server.py`
3. Register a SIP phone to FreeSWITCH (use extension 1001, password 1234)
4. Call extension 1000

## Debugging
```bash
# FreeSWITCH console
docker exec -it freeswitch fs_cli

# Check WebSocket connection
sofia status
show channels

# Enable debug logging
sofia loglevel all 9
```
