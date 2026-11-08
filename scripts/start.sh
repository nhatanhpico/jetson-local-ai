#!/usr/bin/env bash
set -euo pipefail
if ! systemctl is-active --quiet ollama; then
  sudo systemctl start ollama
fi
systemctl --user start local-ai
echo "Local: http://127.0.0.1:7860"
echo "LAN addresses (only reachable if LOCAL_AI_HOST=0.0.0.0 is set in the service):"
ip -4 -brief address
