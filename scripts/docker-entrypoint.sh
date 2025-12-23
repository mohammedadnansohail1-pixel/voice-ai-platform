#!/bin/bash
set -e

echo "=========================================="
echo "Voice AI Platform - Starting"
echo "=========================================="

# Wait for Ollama to be ready
echo "Waiting for Ollama..."
until curl -sf http://${OLLAMA_HOST:-ollama:11434}/ > /dev/null 2>&1; do
    echo "  Ollama not ready, waiting..."
    sleep 2
done
echo "Ollama is ready!"

# Update config with environment variables
if [ -n "$VP_LLM_BASE_URL" ]; then
    echo "LLM URL: $VP_LLM_BASE_URL"
fi

# Start server
echo "Starting Voice Platform..."
exec python scripts/run_server.py
