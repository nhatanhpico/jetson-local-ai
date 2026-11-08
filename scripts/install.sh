#!/usr/bin/env bash
# One-shot setup for a Jetson Orin (JetPack 6) with Ollama already installed.
# Re-runnable: every step skips work that is already done.
set -euo pipefail
AI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAT_MODEL="${LOCAL_AI_CHAT_MODEL:-huihui_ai/qwen3-vl-abliterated:4b-instruct}"
SD_URL=https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors

command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com/download"; exit 1; }
sudo apt-get update && sudo apt-get install -y cmake build-essential python3-venv git

[ -d "$AI_ROOT/venv" ] || python3 -m venv "$AI_ROOT/venv"
"$AI_ROOT/venv/bin/pip" install -r "$AI_ROOT/requirements.lock.txt"

ollama pull "$CHAT_MODEL"
ollama pull "${LOCAL_AI_EMBED_MODEL:-nomic-embed-text}"   # 274 MB, used by long-term memory (Mem0)
mkdir -p "$AI_ROOT/models" "$AI_ROOT/outputs"
[ -f "$AI_ROOT/models/sd15.safetensors" ] || curl -L --fail --retry 3 -o "$AI_ROOT/models/sd15.safetensors" "$SD_URL"

[ -x "$AI_ROOT/stable-diffusion.cpp/build/bin/sd-cli" ] || bash "$AI_ROOT/scripts/rebuild-image-engine.sh"

# systemd user service (generated for this checkout path and user)
mkdir -p "$HOME/.config/systemd/user"
sed "s|@AI_ROOT@|$AI_ROOT|g" "$AI_ROOT/scripts/local-ai.service" > "$HOME/.config/systemd/user/local-ai.service"
systemctl --user daemon-reload
systemctl --user enable --now local-ai
echo "Done. Open http://127.0.0.1:7860"
