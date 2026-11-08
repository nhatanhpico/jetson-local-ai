#!/usr/bin/env bash
set -euo pipefail
systemctl --user stop local-ai
ollama stop "${LOCAL_AI_CHAT_MODEL:-huihui_ai/qwen3-vl-abliterated:4b-instruct}" || true
