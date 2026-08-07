#!/bin/bash
set -e

echo "---Starting Ollama server in background---"
ollama serve &

echo "---Waiting for Ollama server to be active---"
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "---Pulling required models---"
ollama pull nomic-embed-text
ollama pull qwen2.5:7b-instruct

echo "---Keeping Ollama server running---"
wait